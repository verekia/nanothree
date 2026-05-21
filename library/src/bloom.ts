// HDR bloom post-process pass.
//
// Pipeline:
//   1. Threshold + downsample: sceneView (HDR) -> mip[0] (HDR)
//   2. Plain downsample chain: mip[i] -> mip[i+1] (3 more times)
//   3. Additive upsample (tent 3x3): mip[i+1] -> mip[i]
//      Each level keeps the contribution of the previous downsample,
//      so the final mip[0] is a sum of progressively blurred scales.
//
// The result is exposed as `outputView` (= mip[0]) for the tone-mapping
// composite pass to read alongside the original scene. This pass does not
// write to the canvas; it never tone-maps. See `tonemap.ts` for both.

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
@group(0) @binding(2) var<uniform> params: vec4f; // x=threshold

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

const BLOOM_MIPS = 4

export class BloomPass {
  enabled = false
  strength = 0.6
  /**
   * HDR brightness above which pixels contribute to bloom. Lit Lambert
   * surfaces typically stay <=1.0; emissive can go well above. Defaults
   * to 1.0 so only emissive blooms by default.
   */
  threshold = 1.0
  /**
   * Multiplier on each upsample step's contribution. Higher values give
   * a wider, softer halo (lower-resolution mips bleed up more strongly);
   * lower values give a tighter, sharper halo (small mips contribute less).
   * Implemented as a per-pass `blendConstant` on the additive upsample.
   */
  radius = 1.0

  private device!: GPUDevice
  private sceneFormat!: GPUTextureFormat
  private width = 0
  private height = 0
  private _lastSceneView: GPUTextureView | null = null

  private sampler!: GPUSampler
  private paramsBuffer!: GPUBuffer
  private paramsStaging = new Float32Array(4)

  private downsampleLayout!: GPUBindGroupLayout // tex + sampler + params (first pass only)
  private plainLayout!: GPUBindGroupLayout // tex + sampler (downsample + upsample)

  private thresholdPipeline!: GPURenderPipeline
  private downsamplePipeline!: GPURenderPipeline
  private upsamplePipeline!: GPURenderPipeline

  private mipTextures: GPUTexture[] = []
  private mipViews: GPUTextureView[] = []

  /** Final bloom output (= mip[0]). Consumed by the tone-mapping pass. */
  get outputView(): GPUTextureView | null {
    return this.mipViews[0] ?? null
  }

  private thresholdBindGroup: GPUBindGroup | null = null
  private downsampleBindGroups: GPUBindGroup[] = []
  private upsampleBindGroups: GPUBindGroup[] = []

  init(device: GPUDevice, sceneFormat: GPUTextureFormat) {
    this.device = device
    this.sceneFormat = sceneFormat

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

    const downsamplePipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [this.downsampleLayout] })
    const plainPipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [this.plainLayout] })

    const thresholdModule = device.createShaderModule({ code: FULLSCREEN_VS + THRESHOLD_DOWNSAMPLE_FS })
    this.thresholdPipeline = device.createRenderPipeline({
      layout: downsamplePipelineLayout,
      vertex: { module: thresholdModule, entryPoint: 'vs' },
      fragment: { module: thresholdModule, entryPoint: 'fs', targets: [{ format: sceneFormat }] },
      primitive: { topology: 'triangle-list' },
    })

    const downsampleModule = device.createShaderModule({ code: FULLSCREEN_VS + DOWNSAMPLE_FS })
    this.downsamplePipeline = device.createRenderPipeline({
      layout: plainPipelineLayout,
      vertex: { module: downsampleModule, entryPoint: 'vs' },
      fragment: { module: downsampleModule, entryPoint: 'fs', targets: [{ format: sceneFormat }] },
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
            format: sceneFormat,
            // src factor is `constant` so the per-pass blendConstant scales
            // the upsample contribution. This is how `radius` is implemented.
            blend: {
              color: { srcFactor: 'constant', dstFactor: 'one', operation: 'add' },
              alpha: { srcFactor: 'constant', dstFactor: 'one', operation: 'add' },
            },
          },
        ],
      },
      primitive: { topology: 'triangle-list' },
    })
  }

  /**
   * Recreate mip chain + bind groups. Called with the current HDR scene view
   * (owned by ToneMappingPass) so the threshold pass can sample it. Cheap when
   * the size and sceneView haven't changed.
   */
  resize(width: number, height: number, sceneView: GPUTextureView) {
    const sizeChanged = width !== this.width || height !== this.height
    const sceneChanged = sceneView !== this._lastSceneView
    if (!sizeChanged && !sceneChanged) return

    if (sizeChanged) {
      this.width = width
      this.height = height
      for (const t of this.mipTextures) t.destroy()
      this.mipTextures = []
      this.mipViews = []
      for (let i = 0; i < BLOOM_MIPS; i++) {
        const w = Math.max(1, width >> (i + 1))
        const h = Math.max(1, height >> (i + 1))
        const t = this.device.createTexture({
          size: [w, h],
          format: this.sceneFormat,
          usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
        })
        this.mipTextures.push(t)
        this.mipViews.push(t.createView())
      }
    }

    this._lastSceneView = sceneView

    this.thresholdBindGroup = this.device.createBindGroup({
      layout: this.downsampleLayout,
      entries: [
        { binding: 0, resource: sceneView },
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
  }

  encode(encoder: GPUCommandEncoder) {
    this.paramsStaging[0] = this.threshold
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
    // existing downsampled value at mip[src-1] is preserved via loadOp:'load',
    // and the upsample pipeline blends `src * blendConstant + dst * 1`. The
    // blendConstant is `radius`, so a smaller value tightens the halo and a
    // larger value spreads it.
    for (let step = 0; step < BLOOM_MIPS - 1; step++) {
      const dst = BLOOM_MIPS - 2 - step
      const pass = encoder.beginRenderPass({
        colorAttachments: [{ view: this.mipViews[dst], loadOp: 'load', storeOp: 'store' }],
      })
      pass.setPipeline(this.upsamplePipeline)
      pass.setBlendConstant({ r: this.radius, g: this.radius, b: this.radius, a: 1 })
      pass.setBindGroup(0, this.upsampleBindGroups[step])
      pass.draw(3)
      pass.end()
    }
  }

  dispose() {
    for (const t of this.mipTextures) t.destroy()
    this.paramsBuffer?.destroy()
  }
}
