/**
 * Generates TypeScript modules that inline the Draco and Basis decoder data
 * as base64 strings. These are loaded lazily via dynamic import() at runtime.
 */
import { readFileSync, writeFileSync } from 'node:fs'
import { resolve } from 'node:path'

const decodersDir = resolve(import.meta.dirname!, '../decoders')
const srcDir = resolve(import.meta.dirname!, '../src')

function toBase64(filePath: string): string {
  return readFileSync(filePath).toString('base64')
}

function readText(filePath: string): string {
  return readFileSync(filePath, 'utf-8')
}

// Draco
const dracoJs = readText(resolve(decodersDir, 'draco/draco_wasm_wrapper_gltf.js'))
const dracoWasm = toBase64(resolve(decodersDir, 'draco/draco_decoder_gltf.wasm'))

writeFileSync(
  resolve(srcDir, 'draco-inline.ts'),
  `// Auto-generated — do not edit. Run \`bun run generate-decoders\` to regenerate.
export const DRACO_JS = ${JSON.stringify(dracoJs)}
export const DRACO_WASM_BASE64 = ${JSON.stringify(dracoWasm)}
`,
)

// Basis
const basisJs = readText(resolve(decodersDir, 'basis/basis_transcoder.js'))
const basisWasm = toBase64(resolve(decodersDir, 'basis/basis_transcoder.wasm'))

writeFileSync(
  resolve(srcDir, 'basis-inline.ts'),
  `// Auto-generated — do not edit. Run \`bun run generate-decoders\` to regenerate.
export const BASIS_JS = ${JSON.stringify(basisJs)}
export const BASIS_WASM_BASE64 = ${JSON.stringify(basisWasm)}
`,
)

console.log('Generated draco-inline.ts and basis-inline.ts')
