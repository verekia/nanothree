// Material classes for nanothree

import { Color } from './core'

export const FrontSide = 0
export const BackSide = 1
export const DoubleSide = 2
export type Side = typeof FrontSide | typeof BackSide | typeof DoubleSide

/**
 * Diffuse shading model on `MeshLambertMaterial`.
 * - `'lambert'` (default): `max(N·L, 0)`. Edges drop to 0 — punchy contrast.
 * - `'half-lambert'`: Valve's `(N·L·0.5 + 0.5)²`. Lifts edge transition to
 *   ~0.25 of full light, giving softer wraparound. Same [0,1] envelope, so a
 *   light budget of `ambient + directional = 1` peaks at white in both modes.
 */
export type Shading = 'lambert' | 'half-lambert'

/** GPU texture wrapper for nanothree. */
export class NanoTexture {
  _gpuTexture: GPUTexture | null = null
  _gpuView: GPUTextureView | null = null
  _device: GPUDevice | null = null
  _dirty = true

  // Materials that hold this texture in their `map` slot. Used so `image`
  // arriving asynchronously (e.g. from `loadTexture`) can flip their cached
  // `hasTexture` flag without each renderer pass having to re-check.
  _materials: MeshLambertMaterial[] = []

  private _image: ImageBitmap | HTMLImageElement | null

  get image(): ImageBitmap | HTMLImageElement | null {
    return this._image
  }
  set image(v: ImageBitmap | HTMLImageElement | null) {
    if (this._image === v) return
    this._image = v
    const hasIt = v !== null
    for (let i = 0; i < this._materials.length; i++) this._materials[i].hasTexture = hasIt
  }

  constructor(image: ImageBitmap | HTMLImageElement | null = null) {
    this._image = image
  }

  _ensureGPU(device: GPUDevice, useWriteTexture = false) {
    if (!this._dirty && this._device === device) return
    if (!this._image) return
    this._device = device

    const w = this._image.width
    const h = this._image.height

    if (this._gpuTexture) this._gpuTexture.destroy()
    this._gpuTexture = device.createTexture({
      size: [w, h],
      format: 'rgba8unorm',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
    })
    this._gpuView = this._gpuTexture.createView()

    if (useWriteTexture) {
      // Pixel 10 (PowerVR img-tec / d-series): copyExternalImageToTexture
      // silently produces an all-zero texture. Rasterise to a 2D canvas and
      // upload raw RGBA bytes via writeTexture instead.
      const canvas =
        typeof OffscreenCanvas !== 'undefined' ? new OffscreenCanvas(w, h) : document.createElement('canvas')
      if (!(canvas instanceof OffscreenCanvas)) {
        canvas.width = w
        canvas.height = h
      }
      const ctx = (canvas as HTMLCanvasElement | OffscreenCanvas).getContext('2d') as
        | CanvasRenderingContext2D
        | OffscreenCanvasRenderingContext2D
        | null
      if (ctx) {
        ctx.drawImage(this._image as CanvasImageSource, 0, 0)
        const pixels = ctx.getImageData(0, 0, w, h).data
        device.queue.writeTexture({ texture: this._gpuTexture }, pixels, { bytesPerRow: w * 4 }, [w, h, 1])
      } else {
        device.queue.copyExternalImageToTexture({ source: this._image }, { texture: this._gpuTexture }, [w, h])
      }
    } else {
      device.queue.copyExternalImageToTexture({ source: this._image }, { texture: this._gpuTexture }, [w, h])
    }
    this._dirty = false
  }

  dispose() {
    this._gpuTexture?.destroy()
    this._gpuTexture = null
    this._gpuView = null
    this._device = null
  }
}

export class MeshLambertMaterial {
  color: Color
  /** Emissive color added to the final fragment output, independent of lighting. */
  emissive: Color
  /** Scalar multiplier on `emissive` before it's added. Defaults to 1. */
  emissiveIntensity: number
  /** Diffuse shading model. See {@link Shading}. Defaults to `'lambert'`. */
  shading: Shading
  wireframe: boolean
  side: Side
  /** When true, per-vertex colors from the geometry are used (multiplied with material color). */
  vertexColors: boolean

  /**
   * Cached "is this material currently textured" flag. Mirrors
   * `this._map !== null && this._map.image !== null` but as a plain field so
   * the per-frame render-classification loop avoids the getter + property
   * chain on every mesh. Kept in sync by the `map` setter and by
   * `NanoTexture.image` notifying its consumers when the bitmap arrives.
   */
  hasTexture: boolean = false

  // GPU bind group for texture (lazily created)
  _textureBindGroup: GPUBindGroup | null = null
  _textureDirty = true

  private _map: NanoTexture | null = null

  /** Albedo/diffuse texture map. When set, texture color is multiplied with material color. */
  get map(): NanoTexture | null {
    return this._map
  }
  set map(v: NanoTexture | null) {
    if (this._map === v) return
    if (this._map) {
      const arr = this._map._materials
      const i = arr.indexOf(this)
      if (i !== -1) arr.splice(i, 1)
    }
    this._map = v
    if (v !== null) {
      v._materials.push(this)
      this.hasTexture = v.image !== null
    } else {
      this.hasTexture = false
    }
    this._textureBindGroup = null
    this._textureDirty = true
  }

  constructor(params?: {
    color?: Color | number
    emissive?: Color | number
    emissiveIntensity?: number
    shading?: Shading
    wireframe?: boolean
    side?: Side
    map?: NanoTexture
    vertexColors?: boolean
  }) {
    if (params?.color instanceof Color) {
      this.color = params.color
    } else if (typeof params?.color === 'number') {
      this.color = new Color(params.color)
    } else {
      this.color = new Color(0xffffff)
    }
    if (params?.emissive instanceof Color) {
      this.emissive = params.emissive
    } else if (typeof params?.emissive === 'number') {
      this.emissive = new Color(params.emissive)
    } else {
      this.emissive = new Color(0, 0, 0)
    }
    this.emissiveIntensity = params?.emissiveIntensity ?? 1
    this.shading = params?.shading ?? 'lambert'
    this.wireframe = params?.wireframe ?? false
    this.side = params?.side ?? FrontSide
    this.vertexColors = params?.vertexColors ?? false
    if (params?.map) this.map = params.map
  }

  dispose() {
    this._map?.dispose()
    this._textureBindGroup = null
  }
}

export class MeshBasicMaterial {
  readonly isBasic = true
  color: Color
  wireframe: boolean
  side: Side
  /** When true, per-vertex colors from the geometry are used (multiplied with material color). */
  vertexColors: boolean

  constructor(params?: { color?: Color | number; wireframe?: boolean; side?: Side; vertexColors?: boolean }) {
    if (params?.color instanceof Color) {
      this.color = params.color
    } else if (typeof params?.color === 'number') {
      this.color = new Color(params.color)
    } else {
      this.color = new Color(0xffffff)
    }
    this.wireframe = params?.wireframe ?? false
    this.side = params?.side ?? FrontSide
    this.vertexColors = params?.vertexColors ?? false
  }

  dispose() {}
}

export class LineBasicMaterial {
  color: Color

  constructor(params?: { color?: Color | number }) {
    if (params?.color instanceof Color) {
      this.color = params.color
    } else if (typeof params?.color === 'number') {
      this.color = new Color(params.color)
    } else {
      this.color = new Color(0xffffff)
    }
  }

  dispose() {}
}
