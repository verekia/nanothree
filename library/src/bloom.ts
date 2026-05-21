// Minimum-viable bloom post-process pass.
//
// Pipeline:
//   1. Threshold + downsample: sceneTexture -> mip[0]
//   2. Plain downsample chain: mip[i] -> mip[i+1] (3 more times)
//   3. Additive upsample (tent 3x3): mip[i+1] -> mip[i]
//      Each level keeps the contribution of the previous downsample,
//      so the final mip[0] is a sum of progressively blurred scales.
//   4. Composite: scene + mip[0] * strength -> canvas
//
// LDR (matches the canvas format). When emissive / HDR lands, swap the
// scene + mip-chain format to rgba16float and add a tone-map step in the
// composite shader.

const FULLSCREEN_VS = /* wgsl */ `
struct VSOut {
  @builtin(position) pos: vec4f,
  @location(0) uv: vec2f,
}

@vertex fn vs(@builtin(vertex_index) vi: u32) -> VSOut {
  let xy = vec2f(f32((vi << 1u) & 2u), f32(vi & 2u));
  var out: VSOut;
  out.pos = vec4f(xy * 2.0 - 1.0, 0.0, 1.0);
  out.uv = vec2f(xy.x, 1.0 - xy.y);
  return out;
}
`

const THRESHOLD_DOWNSAMPLE_FS = /* wgsl */ `
@group(0) @binding(0) var srcTex: texture_2d<f32>;
@group(0) @binding(1) var srcSampler: sampler;
@group(0) @binding(2) var<uniform> params: vec4f; // x=threshold, y=strength

@fragment fn fs(in: VSOut) -> @location(0) vec4f {
  let texel = 1.0 / vec2f(textureDimensions(srcTex, 0));
  let o = texel * 0.5;
  let c0 = textureSample(srcTex, srcSampler, in.uv + vec2f(-o.x, -o.y)).rgb;
  let c1 = textureSample(srcTex, srcSampler, in.uv + vec2f( o.x, -o.y)).rgb;
  let c2 = textureSample(srcTex, srcSampler, in.uv + vec2f(-o.x,  o.y)).rgb;
  let c3 = textureSample(srcTex, srcSampler, in.uv + vec2f( o.x,  o.y)).rgb;
  let c = (c0 + c1 + c2 + c3) * 0.25;
  let brightness = max(c.r, max(c.g, c.b));
  let contribution = max(brightness - params.x, 0.0) / max(brightness, 1e-4);
  return vec4f(c * contribution, 1.0);
}
`

const DOWNSAMPLE_FS = /* wgsl */ `
@group(0) @binding(0) var srcTex: texture_2d<f32>;
@group(0) @binding(1) var srcSampler: sampler;

@fragment fn fs(in: VSOut) -> @location(0) vec4f {
  let texel = 1.0 / vec2f(textureDimensions(srcTex, 0));
  let o = texel * 0.5;
  let c0 = textureSample(srcTex, srcSampler, in.uv + vec2f(-o.x, -o.y)).rgb;
  let c1 = textureSample(srcTex, srcSampler, in.uv + vec2f( o.x, -o.y)).rgb;
  let c2 = textureSample(srcTex, srcSampler, in.uv + vec2f(-o.x,  o.y)).rgb;
  let c3 = textureSample(srcTex, srcSampler, in.uv + vec2f( o.x,  o.y)).rgb;
  return vec4f((c0 + c1 + c2 + c3) * 0.25, 1.0);
}
`

const UPSAMPLE_FS = /* wgsl */ `
@group(0) @binding(0) var srcTex: texture_2d<f32>;
@group(0) @binding(1) var srcSampler: sampler;

@fragment fn fs(in: VSOut) -> @location(0) vec4f {
  let texel = 1.0 / vec2f(textureDimensions(srcTex, 0));
  let d = texel;
  let s =
    textureSample(srcTex, srcSampler, in.uv + vec2f(-d.x, -d.y)).rgb * 1.0 +
    textureSample(srcTex, srcSampler, in.uv + vec2f( 0.0, -d.y)).rgb * 2.0 +
    textureSample(srcTex, srcSampler, in.uv + vec2f( d.x, -d.y)).rgb * 1.0 +
    textureSample(srcTex, srcSampler, in.uv + vec2f(-d.x,  0.0)).rgb * 2.0 +
    textureSample(srcTex, srcSampler, in.uv                    ).rgb * 4.0 +
    textureSample(srcTex, srcSampler, in.uv + vec2f( d.x,  0.0)).rgb * 2.0 +
    textureSample(srcTex, srcSampler, in.uv + vec2f(-d.x,  d.y)).rgb * 1.0 +
    textureSample(srcTex, srcSampler, in.uv + vec2f( 0.0,  d.y)).rgb * 2.0 +
    textureSample(srcTex, srcSampler, in.uv + vec2f( d.x,  d.y)).rgb * 1.0;
  return vec4f(s * (1.0 / 16.0), 1.0);
}
`

const COMPOSITE_FS = /* wgsl */ `
@group(0) @binding(0) var sceneTex: texture_2d<f32>;
@group(0) @binding(1) var bloomTex: texture_2d<f32>;
@group(0) @binding(2) var smp: sampler;
@group(0) @binding(3) var<uniform> params: vec4f; // x=threshold, y=strength

@fragment fn fs(in: VSOut) -> @location(0) vec4f {
  let scene = textureSample(sceneTex, smp, in.uv).rgb;
  let bloom = textureSample(bloomTex, smp, in.uv).rgb;
  return vec4f(scene + bloom * params.y, 1.0);
}
`

const BLOOM_MIPS = 4

export class BloomPass {
  enabled = false
  strength = 0.6
  threshold = 0.85

  private device!: GPUDevice
  private format!: GPUTextureFormat
  private width = 0
  private height = 0

  private sampler!: GPUSampler
  private paramsBuffer!: GPUBuffer
  private paramsStaging = new Float32Array(4)

  private downsampleLayout!: GPUBindGroupLayout // tex + sampler + params (first pass only)
  private plainLayout!: GPUBindGroupLayout // tex + sampler (downsample + upsample)
  private compositeLayout!: GPUBindGroupLayout

  private thresholdPipeline!: GPURenderPipeline
  private downsamplePipeline!: GPURenderPipeline
  private upsamplePipeline!: GPURenderPipeline
  private compositePipeline!: GPURenderPipeline

  // Resize with canvas
  private sceneTexture: GPUTexture | null = null
  sceneView: GPUTextureView | null = null
  private mipTextures: GPUTexture[] = []
  private mipViews: GPUTextureView[] = []

  private thresholdBindGroup: GPUBindGroup | null = null
  private downsampleBindGroups: GPUBindGroup[] = []
  private upsampleBindGroups: GPUBindGroup[] = []
  private compositeBindGroup: GPUBindGroup | null = null

  init(device: GPUDevice, format: GPUTextureFormat) {
    this.device = device
    this.format = format

    this.sampler = device.createSampler({
      magFilter: 'linear',
      minFilter: 'linear',
      addressModeU: 'clamp-to-edge',
      addressModeV: 'clamp-to-edge',
    })

    this.paramsBuffer = device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })

    this.downsampleLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      ],
    })
    this.plainLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
      ],
    })
    this.compositeLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
        { binding: 3, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      ],
    })

    const downsamplePipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [this.downsampleLayout] })
    const plainPipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [this.plainLayout] })
    const compositePipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [this.compositeLayout] })

    const thresholdModule = device.createShaderModule({ code: FULLSCREEN_VS + THRESHOLD_DOWNSAMPLE_FS })
    this.thresholdPipeline = device.createRenderPipeline({
      layout: downsamplePipelineLayout,
      vertex: { module: thresholdModule, entryPoint: 'vs' },
      fragment: { module: thresholdModule, entryPoint: 'fs', targets: [{ format }] },
      primitive: { topology: 'triangle-list' },
    })

    const downsampleModule = device.createShaderModule({ code: FULLSCREEN_VS + DOWNSAMPLE_FS })
    this.downsamplePipeline = device.createRenderPipeline({
      layout: plainPipelineLayout,
      vertex: { module: downsampleModule, entryPoint: 'vs' },
      fragment: { module: downsampleModule, entryPoint: 'fs', targets: [{ format }] },
      primitive: { topology: 'triangle-list' },
    })

    const upsampleModule = device.createShaderModule({ code: FULLSCREEN_VS + UPSAMPLE_FS })
    this.upsamplePipeline = device.createRenderPipeline({
      layout: plainPipelineLayout,
      vertex: { module: upsampleModule, entryPoint: 'vs' },
      fragment: {
        module: upsampleModule,
        entryPoint: 'fs',
        targets: [
          {
            format,
            blend: {
              color: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
              alpha: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
            },
          },
        ],
      },
      primitive: { topology: 'triangle-list' },
    })

    const compositeModule = device.createShaderModule({ code: FULLSCREEN_VS + COMPOSITE_FS })
    this.compositePipeline = device.createRenderPipeline({
      layout: compositePipelineLayout,
      vertex: { module: compositeModule, entryPoint: 'vs' },
      fragment: { module: compositeModule, entryPoint: 'fs', targets: [{ format }] },
      primitive: { topology: 'triangle-list' },
    })
  }

  resize(width: number, height: number) {
    if (width === this.width && height === this.height) return
    this.width = width
    this.height = height

    this.sceneTexture?.destroy()
    for (const t of this.mipTextures) t.destroy()
    this.mipTextures = []
    this.mipViews = []

    this.sceneTexture = this.device.createTexture({
      size: [width, height],
      format: this.format,
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    })
    this.sceneView = this.sceneTexture.createView()

    for (let i = 0; i < BLOOM_MIPS; i++) {
      const w = Math.max(1, width >> (i + 1))
      const h = Math.max(1, height >> (i + 1))
      const t = this.device.createTexture({
        size: [w, h],
        format: this.format,
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
      })
      this.mipTextures.push(t)
      this.mipViews.push(t.createView())
    }

    this.thresholdBindGroup = this.device.createBindGroup({
      layout: this.downsampleLayout,
      entries: [
        { binding: 0, resource: this.sceneView },
        { binding: 1, resource: this.sampler },
        { binding: 2, resource: { buffer: this.paramsBuffer } },
      ],
    })

    this.downsampleBindGroups = []
    for (let i = 0; i < BLOOM_MIPS - 1; i++) {
      this.downsampleBindGroups.push(
        this.device.createBindGroup({
          layout: this.plainLayout,
          entries: [
            { binding: 0, resource: this.mipViews[i] },
            { binding: 1, resource: this.sampler },
          ],
        }),
      )
    }

    this.upsampleBindGroups = []
    for (let i = BLOOM_MIPS - 1; i > 0; i--) {
      this.upsampleBindGroups.push(
        this.device.createBindGroup({
          layout: this.plainLayout,
          entries: [
            { binding: 0, resource: this.mipViews[i] },
            { binding: 1, resource: this.sampler },
          ],
        }),
      )
    }

    this.compositeBindGroup = this.device.createBindGroup({
      layout: this.compositeLayout,
      entries: [
        { binding: 0, resource: this.sceneView },
        { binding: 1, resource: this.mipViews[0] },
        { binding: 2, resource: this.sampler },
        { binding: 3, resource: { buffer: this.paramsBuffer } },
      ],
    })
  }

  encode(encoder: GPUCommandEncoder, canvasView: GPUTextureView) {
    this.paramsStaging[0] = this.threshold
    this.paramsStaging[1] = this.strength
    this.device.queue.writeBuffer(this.paramsBuffer, 0, this.paramsStaging as unknown as ArrayBuffer)

    // Threshold downsample: scene -> mip[0]
    {
      const pass = encoder.beginRenderPass({
        colorAttachments: [
          { view: this.mipViews[0], clearValue: { r: 0, g: 0, b: 0, a: 1 }, loadOp: 'clear', storeOp: 'store' },
        ],
      })
      pass.setPipeline(this.thresholdPipeline)
      pass.setBindGroup(0, this.thresholdBindGroup!)
      pass.draw(3)
      pass.end()
    }

    // Plain downsample: mip[i] -> mip[i+1]
    for (let i = 0; i < BLOOM_MIPS - 1; i++) {
      const pass = encoder.beginRenderPass({
        colorAttachments: [
          { view: this.mipViews[i + 1], clearValue: { r: 0, g: 0, b: 0, a: 1 }, loadOp: 'clear', storeOp: 'store' },
        ],
      })
      pass.setPipeline(this.downsamplePipeline)
      pass.setBindGroup(0, this.downsampleBindGroups[i])
      pass.draw(3)
      pass.end()
    }

    // Additive upsample: mip[src] blurred and added into mip[src-1]. The
    // existing downsampled value at mip[src-1] is preserved via loadOp:'load'
    // and the upsample pipeline blends src*1 + dst*1.
    for (let step = 0; step < BLOOM_MIPS - 1; step++) {
      const dst = BLOOM_MIPS - 2 - step
      const pass = encoder.beginRenderPass({
        colorAttachments: [{ view: this.mipViews[dst], loadOp: 'load', storeOp: 'store' }],
      })
      pass.setPipeline(this.upsamplePipeline)
      pass.setBindGroup(0, this.upsampleBindGroups[step])
      pass.draw(3)
      pass.end()
    }

    // Composite scene + mip[0] -> canvas
    {
      const pass = encoder.beginRenderPass({
        colorAttachments: [
          { view: canvasView, clearValue: { r: 0, g: 0, b: 0, a: 1 }, loadOp: 'clear', storeOp: 'store' },
        ],
      })
      pass.setPipeline(this.compositePipeline)
      pass.setBindGroup(0, this.compositeBindGroup!)
      pass.draw(3)
      pass.end()
    }
  }

  dispose() {
    this.sceneTexture?.destroy()
    for (const t of this.mipTextures) t.destroy()
    this.paramsBuffer?.destroy()
  }
}
