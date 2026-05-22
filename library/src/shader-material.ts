// Custom WGSL shader material for nanothree
//
// The renderer auto-prepends PREAMBLE before your code, giving you access to:
//   scene.viewProj, scene.lightDir, scene.ambient, scene.lightColor
//   objectData.model, objectData.color
//
// Your code provides @vertex fn vs(...) and @fragment fn fs(...).
// Use @group(2) for your own custom uniforms.
//
// Vertex inputs are always:
//   @location(0) position: vec3f
//   @location(1) normal: vec3f
//
// The renderer also auto-wraps your `fs` so its output passes through
// `applyP3Boost`. If you need the helper for explicit use, it is exposed
// in the preamble as `applyP3Boost(c: vec3f, boost: f32) -> vec3f`.

import { Color } from './core'

// Approximately perceptually uniform saturation boost into the Display-P3
// gamut. The input is treated as sRGB-encoded (the renderer's gamma-incorrect
// convention — Lambert math is done on encoded values and written straight to
// a non-srgb canvas). Output is the encoded value to write into a `display-p3`
// canvas. Boost is in [0, 1]: 0 returns the input unchanged (the sRGB fast
// path), 1 scales chroma by 1.5 in OKLab (strong but mostly in-gamut).
//
// Pipeline: sRGB encoded -> EOTF decode (gamma 2.2) -> linear sRGB ->
// OKLab LMS -> cube root -> Lab -> scale (a, b) -> inverse to LMS -> cube ->
// linear sRGB -> linear Display-P3 -> hue-preserving gamut clip ->
// EOTF encode -> P3 encoded.
//
// The EOTF roundtrip keeps boost ≈ 0 colorimetrically identical to the sRGB
// fast path (skip it and the linear-domain matrix amplifies chroma on
// gamma-encoded values, making low boosts look much deeper than they should).
//
// The gamut clip is hue-preserving: instead of clamping linear-P3 channels
// independently (which slides oranges toward red and magentas toward purple
// once chroma is pushed past the P3 edge), we bisect the chroma scale in
// OKLab between 1.0 (the colorimetric P3 of the input — guaranteed in-gamut
// since sRGB ⊂ P3) and the requested scale. Six iterations land within
// ~1.5% of the gamut edge while keeping L and hue fixed.
export const P3_BOOST_WGSL = /* wgsl */ `
fn _p3b_oklab_chroma_to_linp3(L: f32, k_l: f32, k_m: f32, k_s: f32, scale: f32) -> vec3f {
  let l_ = L + scale * k_l;
  let m_ = L + scale * k_m;
  let s_ = L + scale * k_s;
  let ll = l_ * l_ * l_;
  let mm = m_ * m_ * m_;
  let ss = s_ * s_ * s_;
  let rs =  4.0767416621 * ll - 3.3077115913 * mm + 0.2309699292 * ss;
  let gs = -1.2684380046 * ll + 2.6097574011 * mm - 0.3413193965 * ss;
  let bs = -0.0041960863 * ll - 0.7034186147 * mm + 1.7076147010 * ss;
  let rp = 0.8224621 * rs + 0.1775380 * gs;
  let gp = 0.0331942 * rs + 0.9668058 * gs;
  let bp = 0.0170828 * rs + 0.0723976 * gs + 0.9105197 * bs;
  return vec3f(rp, gp, bp);
}

fn _p3b_in_gamut(c: vec3f) -> bool {
  let lo = min(min(c.r, c.g), c.b);
  let hi = max(max(c.r, c.g), c.b);
  return lo >= 0.0 && hi <= 1.0;
}

fn applyP3Boost(c: vec3f, boost: f32) -> vec3f {
  if (boost <= 0.0) { return c; }
  let lin = pow(max(c, vec3f(0.0)), vec3f(2.2));
  let l = 0.4122214708 * lin.r + 0.5363325363 * lin.g + 0.0514459929 * lin.b;
  let m = 0.2119034982 * lin.r + 0.6806995451 * lin.g + 0.1073969566 * lin.b;
  let s = 0.0883024619 * lin.r + 0.2817188376 * lin.g + 0.6299787005 * lin.b;
  let l_ = sign(l) * pow(abs(l), 1.0 / 3.0);
  let m_ = sign(m) * pow(abs(m), 1.0 / 3.0);
  let s_ = sign(s) * pow(abs(s), 1.0 / 3.0);
  let L  = 0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_;
  let a0 = 1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_;
  let b0 = 0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_;
  // Per-unit-scale LMS' deltas along the constant-hue line from gray (L, 0, 0).
  let k_l =  0.3963377774 * a0 + 0.2158037573 * b0;
  let k_m = -0.1055613458 * a0 - 0.0638541728 * b0;
  let k_s = -0.0894841775 * a0 - 1.2914855480 * b0;
  let tgt = 1.0 + boost * 0.5;
  let p3_target = _p3b_oklab_chroma_to_linp3(L, k_l, k_m, k_s, tgt);
  var p3 = p3_target;
  if (!_p3b_in_gamut(p3_target)) {
    // Bisect chroma scale in [1, tgt] to find the largest still inside P3.
    // sRGB ⊂ P3 guarantees scale = 1 is in-gamut, so the search is well-formed.
    var lo = 1.0;
    var hi = tgt;
    var best = _p3b_oklab_chroma_to_linp3(L, k_l, k_m, k_s, 1.0);
    for (var i = 0; i < 6; i = i + 1) {
      let mid = 0.5 * (lo + hi);
      let test = _p3b_oklab_chroma_to_linp3(L, k_l, k_m, k_s, mid);
      if (_p3b_in_gamut(test)) {
        lo = mid;
        best = test;
      } else {
        hi = mid;
      }
    }
    p3 = best;
  }
  // Tiny numerical excess from the bisection rounding — safe to clamp now,
  // chroma is already at the boundary so this can't shift hue.
  let p3lin = clamp(p3, vec3f(0.0), vec3f(1.0));
  return pow(p3lin, vec3f(1.0 / 2.2));
}
`

// Single source of truth for the per-frame Scene UBO. The renderer's
// `sceneData` write must match this layout byte-for-byte; both the
// built-in shaders (renderer.ts) and user ShaderMaterials import this
// to avoid silent drift between the two.
export const SCENE_STRUCT_WGSL = /* wgsl */ `
struct Scene {
  viewProj: mat4x4f,
  lightDir: vec4f,
  ambient: vec4f,
  lightColor: vec4f,
  lightViewProj: mat4x4f,
  shadowParams: vec4f,  // x: enabled, y: bias, z: texelSize
  cameraRight: vec4f,
  cameraUp: vec4f,
  p3Boost: vec4f,       // x: P3 saturation boost amount (0..1)
}

@group(0) @binding(0) var<uniform> scene: Scene;
@group(0) @binding(1) var shadowMap: texture_depth_2d;
@group(0) @binding(2) var shadowSampler: sampler_comparison;
`

export const SHADER_PREAMBLE =
  SCENE_STRUCT_WGSL +
  /* wgsl */ `
struct ObjectData {
  model: mat4x4f,
  color: vec4f,
}

@group(1) @binding(0) var<storage, read> objectData: ObjectData;
` +
  P3_BOOST_WGSL

export interface ShaderMaterialParams {
  code: string
  uniforms?: Float32Array
  color?: Color | number
  wireframe?: boolean
}

// Strips `//` line comments and `/* */` block comments. WGSL has no string
// literals, so this is straightforward — and necessary because the parser
// below uses `indexOf('@fragment fn fs')`, which would otherwise be fooled
// by that string appearing inside a comment.
function stripWgslComments(s: string): string {
  let out = ''
  let i = 0
  while (i < s.length) {
    if (s[i] === '/' && s[i + 1] === '/') {
      i += 2
      while (i < s.length && s[i] !== '\n') i++
    } else if (s[i] === '/' && s[i + 1] === '*') {
      i += 2
      while (i < s.length && !(s[i] === '*' && s[i + 1] === '/')) i++
      i += 2
    } else {
      out += s[i]
      i++
    }
  }
  return out
}

// Splits a comma-separated WGSL parameter list at top-level commas only
// (skips commas inside attribute parens, generics, etc.).
function splitTopLevelCommas(s: string): string[] {
  const out: string[] = []
  let depth = 0
  let start = 0
  for (let i = 0; i < s.length; i++) {
    const ch = s[i]
    if (ch === '(' || ch === '<' || ch === '[') depth++
    else if (ch === ')' || ch === '>' || ch === ']') depth--
    else if (ch === ',' && depth === 0) {
      out.push(s.slice(start, i))
      start = i + 1
    }
  }
  out.push(s.slice(start))
  return out
}

// Rewrites the user's `@fragment fn fs(...) -> @location(0) vec4f { ... }`
// into a plain `fn _p3b_fs_inner_(...) -> vec4f { ... }` and appends a new
// `fs` entry point that pipes the result through `applyP3Boost`. If the
// signature doesn't match (e.g. multi-target output, struct return), the
// code is returned untouched and the boost has no effect on that shader.
function wrapFragmentForP3Boost(rawCode: string): string {
  const code = stripWgslComments(rawCode)
  const tag = '@fragment fn fs'
  const tagIdx = code.indexOf(tag)
  if (tagIdx === -1) return rawCode

  let i = tagIdx + tag.length
  while (i < code.length && code[i] !== '(') i++
  if (i >= code.length) return rawCode
  const argStart = i + 1

  let depth = 1
  i = argStart
  while (i < code.length && depth > 0) {
    if (code[i] === '(') depth++
    else if (code[i] === ')') depth--
    i++
  }
  if (depth !== 0) return rawCode
  const argEnd = i - 1
  const argsStr = code.slice(argStart, argEnd)

  const after = code.slice(i)
  const retMatch = /^\s*->\s*@location\(0\)\s*vec4f\s*\{/.exec(after)
  if (!retMatch) return rawCode
  const bodyStart = i + retMatch[0].length

  const argNames = splitTopLevelCommas(argsStr)
    .map(arg => {
      const colon = arg.indexOf(':')
      if (colon === -1) return ''
      const before = arg.slice(0, colon).trim()
      const m = /(\w+)\s*$/.exec(before)
      return m ? m[1] : ''
    })
    .filter(Boolean)

  const renamedHeader = `fn _p3b_fs_inner_(${argsStr}) -> vec4f {`
  const wrapper = `

@fragment fn fs(${argsStr}) -> @location(0) vec4f {
  let _p3b_c = _p3b_fs_inner_(${argNames.join(', ')});
  return vec4f(applyP3Boost(_p3b_c.rgb, scene.p3Boost.x), _p3b_c.a);
}
`
  return code.slice(0, tagIdx) + renamedHeader + code.slice(bodyStart) + wrapper
}

export class ShaderMaterial {
  readonly isShaderMaterial = true

  color: Color
  wireframe: boolean
  readonly code: string
  readonly uniforms: Float32Array | null

  // GPU resources managed by the renderer
  _uniformBuffer: GPUBuffer | null = null
  _uniformBindGroup: GPUBindGroup | null = null
  _device: GPUDevice | null = null

  constructor(params: ShaderMaterialParams) {
    this.code = params.code
    this.uniforms = params.uniforms ?? null
    this.wireframe = params.wireframe ?? false

    if (params.color instanceof Color) {
      this.color = params.color
    } else if (typeof params.color === 'number') {
      this.color = new Color(params.color)
    } else {
      this.color = new Color(1, 1, 1)
    }
  }

  // Full WGSL: preamble + user code with `fs` auto-wrapped for the P3 boost.
  get fullCode(): string {
    return SHADER_PREAMBLE + wrapFragmentForP3Boost(this.code)
  }

  // Cache key for pipeline lookup (code content + uniforms presence)
  get _cacheKey(): string {
    return this.code + (this.uniforms ? '\0:u' : '\0:n') + (this.wireframe ? '\0:w' : '\0:s')
  }

  _ensureGPU(device: GPUDevice, uniformLayout: GPUBindGroupLayout) {
    if (!this.uniforms) return

    if (!this._uniformBuffer || this._device !== device) {
      this._device = device
      if (this._uniformBuffer) this._uniformBuffer.destroy()

      // Pad to 16 bytes minimum (WebGPU requirement)
      const size = Math.max(this.uniforms.byteLength, 16)
      this._uniformBuffer = device.createBuffer({
        size,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      })
      this._uniformBindGroup = device.createBindGroup({
        layout: uniformLayout,
        entries: [{ binding: 0, resource: { buffer: this._uniformBuffer } }],
      })
    }

    // Upload uniform data every frame (user mutates the Float32Array directly)
    device.queue.writeBuffer(this._uniformBuffer, 0, this.uniforms as unknown as ArrayBuffer)
  }

  dispose() {
    this._uniformBuffer?.destroy()
    this._uniformBuffer = null
    this._uniformBindGroup = null
    this._device = null
  }
}
