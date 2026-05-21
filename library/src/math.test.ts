import { describe, expect, test } from 'bun:test'

import { mat4ComposeTRS, mat4Identity } from './math'

describe('math', () => {
  test('mat4Identity creates identity matrix', () => {
    const m = new Float32Array(16)
    mat4Identity(m)
    expect(m[0]).toBe(1)
    expect(m[5]).toBe(1)
    expect(m[10]).toBe(1)
    expect(m[15]).toBe(1)
    expect(m[1]).toBe(0)
  })

  test('mat4ComposeTRS Y-only fast path matches general formula', () => {
    // Y-only fast path triggers when rx === 0 && rz === 0.
    const ry = 0.7
    const fast = new Float32Array(16)
    mat4ComposeTRS(fast, 1, 2, 3, 0, ry, 0, 1.5, 2, 0.5)
    const cy = Math.cos(ry)
    const sy2 = Math.sin(ry)
    expect(fast[0]).toBeCloseTo(cy * 1.5)
    expect(fast[1]).toBe(0)
    expect(fast[2]).toBeCloseTo(-sy2 * 1.5)
    expect(fast[4]).toBe(0)
    expect(fast[5]).toBe(2)
    expect(fast[6]).toBe(0)
    expect(fast[8]).toBeCloseTo(sy2 * 0.5)
    expect(fast[9]).toBe(0)
    expect(fast[10]).toBeCloseTo(cy * 0.5)
    expect(fast[12]).toBe(1)
    expect(fast[13]).toBe(2)
    expect(fast[14]).toBe(3)
    expect(fast[15]).toBe(1)
  })

  test('mat4ComposeTRS general path produces same output as Y-only when rx=rz=0', () => {
    // Force both paths to produce identical output for a Y-only rotation by
    // perturbing rx/rz with a tiny non-zero so the general path runs, then
    // compare against the fast-path output.
    const ry = 1.234
    const sx = 2,
      sy = 3,
      sz = 0.7
    const fast = new Float32Array(16)
    mat4ComposeTRS(fast, 0, 0, 0, 0, ry, 0, sx, sy, sz)
    const general = new Float32Array(16)
    // Use 1e-300 so it survives the branch but contributes negligible to math.
    mat4ComposeTRS(general, 0, 0, 0, 1e-300, ry, 1e-300, sx, sy, sz)
    for (let i = 0; i < 16; i++) expect(fast[i]).toBeCloseTo(general[i])
  })
})
