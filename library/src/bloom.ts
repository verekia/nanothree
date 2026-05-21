// HDR bloom post-process pass.
//
// Pipeline:
//   1. Threshold + downsample: sceneTexture (HDR) -> mip[0] (HDR)
//   2. Plain downsample chain: mip[i] -> mip[i+1] (3 more times)
//   3. Additive upsample (tent 3x3): mip[i+1] -> mip[i]
//      Each level keeps the contribution of the previous downsample,
//      so the final mip[0] is a sum of progressively blurred scales.
//   4. Composite: tone-map(scene + mip[0] * strength) -> canvas (LDR)
//
// The scene + mip chain live in `sceneFormat` (HDR rgba16float). The composite
// pipeline writes to `outputFormat` (canvas LDR).
//
// Tone-mapping is opt-in via `BloomPass.toneMapping`. Default `NoToneMapping`
// hard-clamps each channel to [0,1] — matches the punchy pre-HDR look. Set to
// `ACESFilmicToneMapping` for a soft shoulder + highlight desaturation.

export const NoToneMapping = 0
export const ACESFilmicToneMapping = 1
export const SoftToneMapping = 2
export const AgXToneMapping = 3
export const NeutralToneMapping = 4
export type ToneMapping =
  | typeof NoToneMapping
  | typeof ACESFilmicToneMapping
  | typeof SoftToneMapping
  | typeof AgXToneMapping
  | typeof NeutralToneMapping

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
@group(0) @binding(3) var<uniform> params: vec4f; // x=threshold, y=strength, z=toneMapping mode

// ACES filmic tone map (Narkowicz approximation). HDR -> LDR with a
// soft shoulder; keeps colored midtones, rolls off highlights to white.
// Crushes mid-range saturation on LDR-input scenes — use for cinematic feel.
fn tonemapACES(c: vec3f) -> vec3f {
  let a = 2.51;
  let b = 0.03;
  let cc = 2.43;
  let d = 0.59;
  let e = 0.14;
  return saturate((c * (a * c + b)) / (c * (cc * c + d) + e));
}

// Soft shoulder above a 0.8 knee. Identity for c <= 0.8, smooth asymptotic
// roll-off above. C0+C1 continuous at the knee so there is no visible seam.
// Best choice when you want LDR-look preserved but emissive >1 to soft-clip.
fn tonemapSoft(c: vec3f) -> vec3f {
  let knee = vec3f(0.8);
  let oneMinusKnee = vec3f(1.0) - knee;
  let above = knee + oneMinusKnee * (c - knee) / ((c - knee) + oneMinusKnee);
  return saturate(select(above, c, c < knee));
}

fn agxContrast(x: vec3f) -> vec3f {
  let x2 = x * x;
  let x4 = x2 * x2;
  return 15.5 * x4 * x2
       - 40.14 * x4 * x
       + 31.96 * x4
       - 6.868 * x2 * x
       + 0.4298 * x2
       + 0.1191 * x
       - 0.00232;
}

// AgX (Sobotka). Modern filmic curve that preserves saturation much better
// than ACES. Uses an LMS-like input matrix, log-encode, sigmoid contrast,
// then an inverse matrix + EOTF. Reference: iolite-engine.com (minimal AgX).
fn tonemapAgX(c: vec3f) -> vec3f {
  let agxMat = mat3x3f(
    vec3f(0.842479062253094, 0.0423282422610123, 0.0423756549057051),
    vec3f(0.0784335999999992, 0.878468636469772, 0.0784336),
    vec3f(0.0792237451477643, 0.0791661274605434, 0.879142973793104),
  );
  let agxInv = mat3x3f(
    vec3f(1.19687900512017, -0.0528968517574562, -0.0529716355144438),
    vec3f(-0.0980208811401368, 1.15190312990417, -0.0980434501171241),
    vec3f(-0.0990297440797205, -0.0989611768448433, 1.15107367264116),
  );
  let minEv = -12.47393;
  let maxEv = 4.026069;

  var v = agxMat * c;
  v = clamp(log2(max(v, vec3f(1e-10))), vec3f(minEv), vec3f(maxEv));
  v = (v - vec3f(minEv)) / (maxEv - minEv);
  v = agxContrast(v);
  v = agxInv * v;
  return saturate(pow(max(v, vec3f(0.0)), vec3f(2.2)));
}

// Khronos PBR Neutral. Designed for "no-grade" game look: identity below
// 0.76, soft compression above with mild highlight desaturation. Preserves
// mid-tone color better than ACES.
fn tonemapNeutral(c: vec3f) -> vec3f {
  let startCompression = 0.76;
  let desaturation = 0.15;
  var col = c;
  let lo = min(col.r, min(col.g, col.b));
  let offset = select(0.04, lo - 6.25 * lo * lo, lo < 0.08);
  col = col - vec3f(offset);
  let peak = max(col.r, max(col.g, col.b));
  if (peak < startCompression) {
    return col;
  }
  let d = 1.0 - startCompression;
  let newPeak = 1.0 - d * d / (peak + d - startCompression);
  col = col * (newPeak / peak);
  let g = 1.0 - 1.0 / (desaturation * (peak - newPeak) + 1.0);
  return mix(col, vec3f(newPeak), vec3f(g));
}

@fragment fn fs(in: VSOut) -> @location(0) vec4f {
  let scene = textureSample(sceneTex, smp, in.uv).rgb;
  let bloomCol = textureSample(bloomTex, smp, in.uv).rgb;
  let combined = scene + bloomCol * params.y;

  // params.z: 0=None, 1=ACES, 2=Soft, 3=AgX, 4=Neutral
  let mode = i32(params.z + 0.5);
  var mapped: vec3f;
  if (mode == 1) {
    mapped = tonemapACES(combined);
  } else if (mode == 2) {
    mapped = tonemapSoft(combined);
  } else if (mode == 3) {
    mapped = tonemapAgX(combined);
  } else if (mode == 4) {
    mapped = tonemapNeutral(combined);
  } else {
    mapped = saturate(combined);
  }
  return vec4f(mapped, 1.0);
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
  /**
   * Tone-mapping applied at the composite step. `NoToneMapping` (default)
   * hard-clamps each channel to [0,1] — preserves the punchy LDR look but
   * loses highlight detail above 1. `ACESFilmicToneMapping` applies a soft
   * shoulder that bleaches highlights to white.
   */
  toneMapping: ToneMapping = NoToneMapping

  private device!: GPUDevice
  private sceneFormat!: GPUTextureFormat // HDR
  private outputFormat!: GPUTextureFormat // canvas LDR
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

  init(device: GPUDevice, sceneFormat: GPUTextureFormat, outputFormat: GPUTextureFormat) {
    this.device = device
    this.sceneFormat = sceneFormat
    this.outputFormat = outputFormat

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

    const compositeModule = device.createShaderModule({ code: FULLSCREEN_VS + COMPOSITE_FS })
    this.compositePipeline = device.createRenderPipeline({
      layout: compositePipelineLayout,
      vertex: { module: compositeModule, entryPoint: 'vs' },
      fragment: { module: compositeModule, entryPoint: 'fs', targets: [{ format: outputFormat }] },
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
      format: this.sceneFormat,
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    })
    this.sceneView = this.sceneTexture.createView()

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
    this.paramsStaging[2] = this.toneMapping
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
