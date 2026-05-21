// Tone-mapping / final-composite post-process pass.
//
// Owns the HDR scene render target and the final pass that maps HDR scene
// colors (plus an optional bloom contribution) down to LDR canvas values.
//
// Whenever any HDR-aware post-process is active (tone mapping OR bloom),
// the renderer draws into `sceneView` (HDR rgba16float) instead of the
// canvas. After bloom (if enabled) has filled its mip[0], the renderer
// calls `encode()` to composite scene + bloom and tone-map to LDR.
//
// When `toneMapping === NoToneMapping` and bloom is off, the renderer
// bypasses this pass entirely and renders straight to the canvas — the
// "zero post-processing" fast path for weak devices.

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

const COMPOSITE_FS = /* wgsl */ `
@group(0) @binding(0) var sceneTex: texture_2d<f32>;
@group(0) @binding(1) var bloomTex: texture_2d<f32>;
@group(0) @binding(2) var smp: sampler;
@group(0) @binding(3) var<uniform> params: vec4f; // x=bloomStrength, y=toneMapping mode

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
  let combined = scene + bloomCol * params.x;

  // params.y: 0=None, 1=ACES, 2=Soft, 3=AgX, 4=Neutral
  let mode = i32(params.y + 0.5);
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

export class ToneMappingPass {
  private device!: GPUDevice
  private sceneFormat!: GPUTextureFormat
  private outputFormat!: GPUTextureFormat
  private width = 0
  private height = 0

  private sampler!: GPUSampler
  private paramsBuffer!: GPUBuffer
  private paramsStaging = new Float32Array(4)

  private compositeLayout!: GPUBindGroupLayout
  private compositePipeline!: GPURenderPipeline

  // 1×1 black texture used when bloom is disabled so the composite shader
  // can keep a single bind group layout with two source textures.
  private fallbackBloomTexture!: GPUTexture
  private fallbackBloomView!: GPUTextureView

  private sceneTexture: GPUTexture | null = null
  sceneView: GPUTextureView | null = null

  private compositeBindGroup: GPUBindGroup | null = null
  private _lastBloomView: GPUTextureView | null = null

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

    this.fallbackBloomTexture = device.createTexture({
      size: [1, 1],
      format: sceneFormat,
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT,
    })
    this.fallbackBloomView = this.fallbackBloomTexture.createView()
    // Clear the fallback once to zero (encoder-less init via a one-shot pass)
    const enc = device.createCommandEncoder()
    const p = enc.beginRenderPass({
      colorAttachments: [
        { view: this.fallbackBloomView, clearValue: { r: 0, g: 0, b: 0, a: 0 }, loadOp: 'clear', storeOp: 'store' },
      ],
    })
    p.end()
    device.queue.submit([enc.finish()])

    this.compositeLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
        { binding: 3, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      ],
    })

    const layout = device.createPipelineLayout({ bindGroupLayouts: [this.compositeLayout] })
    const module = device.createShaderModule({ code: FULLSCREEN_VS + COMPOSITE_FS })
    this.compositePipeline = device.createRenderPipeline({
      layout,
      vertex: { module, entryPoint: 'vs' },
      fragment: { module, entryPoint: 'fs', targets: [{ format: outputFormat }] },
      primitive: { topology: 'triangle-list' },
    })
  }

  /** Recreate the HDR scene target at the given size. No-op if unchanged. */
  resize(width: number, height: number) {
    if (width === this.width && height === this.height && this.sceneTexture) return
    this.width = width
    this.height = height
    this.sceneTexture?.destroy()
    this.sceneTexture = this.device.createTexture({
      size: [width, height],
      format: this.sceneFormat,
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    })
    this.sceneView = this.sceneTexture.createView()
    // Scene view changed — composite bind group must be rebuilt.
    this.compositeBindGroup = null
    this._lastBloomView = null
  }

  encode(
    encoder: GPUCommandEncoder,
    canvasView: GPUTextureView,
    bloomView: GPUTextureView | null,
    bloomStrength: number,
    toneMapping: ToneMapping,
  ) {
    const bloom = bloomView ?? this.fallbackBloomView
    if (this.compositeBindGroup === null || bloom !== this._lastBloomView) {
      this._lastBloomView = bloom
      this.compositeBindGroup = this.device.createBindGroup({
        layout: this.compositeLayout,
        entries: [
          { binding: 0, resource: this.sceneView! },
          { binding: 1, resource: bloom },
          { binding: 2, resource: this.sampler },
          { binding: 3, resource: { buffer: this.paramsBuffer } },
        ],
      })
    }

    this.paramsStaging[0] = bloomView ? bloomStrength : 0
    this.paramsStaging[1] = toneMapping
    this.device.queue.writeBuffer(this.paramsBuffer, 0, this.paramsStaging as unknown as ArrayBuffer)

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

  dispose() {
    this.sceneTexture?.destroy()
    this.fallbackBloomTexture?.destroy()
    this.paramsBuffer?.destroy()
  }
}
