// Draco WASM decoder for nanothree GLTF loader
//
// Loads the Draco gltf decoder (WASM variant) on first use and caches it.
// By default uses inline decoder data bundled with nanothree. If a custom
// path is set via GLTFLoader.setDracoDecoderPath(), fetches from that URL instead.

// ── Draco module types ───────────────────────────────────────────────

interface DracoModule {
  DecoderBuffer: new () => DracoDecoderBuffer
  Decoder: new () => DracoDecoder
  Mesh: new () => DracoMesh
  PointCloud: new () => DracoPointCloud
  TRIANGULAR_MESH: number
  DT_FLOAT32: number
  HEAPF32: Float32Array
  HEAPU32: Uint32Array
  _malloc(size: number): number
  _free(ptr: number): void
  destroy(obj: unknown): void
}

interface DracoDecoderBuffer {
  Init(data: Int8Array, length: number): void
}

interface DracoDecoder {
  GetEncodedGeometryType(buffer: DracoDecoderBuffer): number
  DecodeBufferToMesh(buffer: DracoDecoderBuffer, mesh: DracoMesh): { ok(): boolean }
  DecodeBufferToPointCloud(buffer: DracoDecoderBuffer, cloud: DracoPointCloud): { ok(): boolean }
  GetAttributeByUniqueId(geo: DracoGeometry, id: number): DracoAttribute | null
  GetTrianglesUInt32Array(mesh: DracoMesh, byteLength: number, ptr: number): void
  GetAttributeDataArrayForAllPoints(
    geo: DracoGeometry,
    attr: DracoAttribute,
    dataType: number,
    byteLength: number,
    ptr: number,
  ): void
}

interface DracoGeometry {
  num_points(): number
}

interface DracoAttribute {
  num_components(): number
}

interface DracoMesh extends DracoGeometry {
  num_faces(): number
}

type DracoPointCloud = DracoGeometry

// ── Helpers ─────────────────────────────────────────────────────────

function base64ToArrayBuffer(base64: string): ArrayBuffer {
  const binary = atob(base64)
  const bytes = new Uint8Array(binary.length)
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i)
  return bytes.buffer as ArrayBuffer
}

// ── Cached module singleton ──────────────────────────────────────────

let cachedModule: DracoModule | null = null
let loadingPromise: Promise<DracoModule> | null = null

async function initDracoModule(jsText: string, wasmBinary: ArrayBuffer): Promise<DracoModule> {
  const fn = new Function('module', 'exports', jsText + '\nreturn DracoDecoderModule;')
  const DracoDecoderModule = fn({}, {})
  return await (DracoDecoderModule({ wasmBinary }) as Promise<DracoModule>)
}

async function loadDracoDecoder(decoderPath: string | null): Promise<DracoModule> {
  if (cachedModule) return cachedModule
  if (loadingPromise) return loadingPromise

  loadingPromise = (async () => {
    let jsText: string
    let wasmBinary: ArrayBuffer

    if (decoderPath) {
      // Fetch from user-specified path
      ;[jsText, wasmBinary] = await Promise.all([
        fetch(decoderPath + 'draco_wasm_wrapper_gltf.js').then(r => r.text()),
        fetch(decoderPath + 'draco_decoder_gltf.wasm').then(r => r.arrayBuffer()),
      ])
    } else {
      // Use bundled inline data (lazy-loaded chunk)
      const { DRACO_JS, DRACO_WASM_BASE64 } = await import('./draco-inline')
      jsText = DRACO_JS
      wasmBinary = base64ToArrayBuffer(DRACO_WASM_BASE64)
    }

    cachedModule = await initDracoModule(jsText, wasmBinary)
    return cachedModule!
  })()

  return loadingPromise
}

// ── Public API ───────────────────────────────────────────────────────

export interface DecodedDracoGeometry {
  positions?: Float32Array
  normals?: Float32Array
  uvs?: Float32Array
  indices?: Uint32Array
}

/**
 * Decode a Draco-compressed mesh from a buffer.
 * @param decoderPath URL path to Draco decoder files (with trailing slash), or null to use bundled decoder
 * @param compressedData The raw compressed bytes from the GLTF bufferView
 * @param dracoAttributes Map of GLTF attribute names to Draco unique IDs from the extension
 */
export async function decodeDracoData(
  decoderPath: string | null,
  compressedData: ArrayBuffer,
  dracoAttributes: Record<string, number>,
): Promise<DecodedDracoGeometry> {
  const module = await loadDracoDecoder(decoderPath)

  const buffer = new module.DecoderBuffer()
  buffer.Init(new Int8Array(compressedData), compressedData.byteLength)

  const decoder = new module.Decoder()
  const geometryType = decoder.GetEncodedGeometryType(buffer)

  let dracoGeometry: DracoMesh | DracoPointCloud
  if (geometryType === module.TRIANGULAR_MESH) {
    dracoGeometry = new module.Mesh()
    const status = decoder.DecodeBufferToMesh(buffer, dracoGeometry as DracoMesh)
    if (!status.ok()) throw new Error('[nanothree] Draco mesh decode failed')
  } else {
    dracoGeometry = new module.PointCloud()
    const status = decoder.DecodeBufferToPointCloud(buffer, dracoGeometry)
    if (!status.ok()) throw new Error('[nanothree] Draco point cloud decode failed')
  }

  const result: DecodedDracoGeometry = {}
  const numPoints = dracoGeometry.num_points()

  // Read each attribute by its Draco unique ID
  for (const [name, uniqueId] of Object.entries(dracoAttributes)) {
    const attr = decoder.GetAttributeByUniqueId(dracoGeometry, uniqueId)
    if (!attr) continue

    const numComponents = attr.num_components()
    const numValues = numComponents * numPoints
    const byteLength = numValues * Float32Array.BYTES_PER_ELEMENT
    const ptr = module._malloc(byteLength)
    decoder.GetAttributeDataArrayForAllPoints(dracoGeometry, attr, module.DT_FLOAT32, byteLength, ptr)
    const data = new Float32Array(module.HEAPF32.buffer, ptr, numValues).slice()
    module._free(ptr)

    if (name === 'POSITION') result.positions = data
    else if (name === 'NORMAL') result.normals = data
    else if (name === 'TEXCOORD_0') result.uvs = data
  }

  // Read triangle indices
  if (geometryType === module.TRIANGULAR_MESH) {
    const mesh = dracoGeometry as DracoMesh
    const numFaces = mesh.num_faces()
    const numIndices = numFaces * 3
    const byteLength = numIndices * Uint32Array.BYTES_PER_ELEMENT
    const ptr = module._malloc(byteLength)
    decoder.GetTrianglesUInt32Array(mesh, byteLength, ptr)
    result.indices = new Uint32Array(module.HEAPU32.buffer, ptr, numIndices).slice()
    module._free(ptr)
  }

  module.destroy(decoder)
  module.destroy(buffer)
  module.destroy(dracoGeometry)
  return result
}
