import { describe, expect, test } from 'bun:test'

import { mat4Identity } from './math'

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
})
