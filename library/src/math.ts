// Lightweight math utilities for nanothree

export function mat4Perspective(out: Float32Array, fov: number, aspect: number, near: number, far: number) {
  const f = 1 / Math.tan(fov / 2)
  const nf = 1 / (near - far)
  out[0] = f / aspect
  out[1] = 0
  out[2] = 0
  out[3] = 0
  out[4] = 0
  out[5] = f
  out[6] = 0
  out[7] = 0
  out[8] = 0
  out[9] = 0
  out[10] = far * nf
  out[11] = -1
  out[12] = 0
  out[13] = 0
  out[14] = near * far * nf
  out[15] = 0
}

export function mat4LookAt(
  out: Float32Array,
  ex: number,
  ey: number,
  ez: number,
  tx: number,
  ty: number,
  tz: number,
  ux: number,
  uy: number,
  uz: number,
) {
  let fx = tx - ex,
    fy = ty - ey,
    fz = tz - ez
  let len = 1 / Math.sqrt(fx * fx + fy * fy + fz * fz)
  fx *= len
  fy *= len
  fz *= len
  let rx = fy * uz - fz * uy,
    ry = fz * ux - fx * uz,
    rz = fx * uy - fy * ux
  len = 1 / Math.sqrt(rx * rx + ry * ry + rz * rz)
  rx *= len
  ry *= len
  rz *= len
  const sx = ry * fz - rz * fy,
    sy = rz * fx - rx * fz,
    sz = rx * fy - ry * fx
  out[0] = rx
  out[1] = sx
  out[2] = -fx
  out[3] = 0
  out[4] = ry
  out[5] = sy
  out[6] = -fy
  out[7] = 0
  out[8] = rz
  out[9] = sz
  out[10] = -fz
  out[11] = 0
  out[12] = -(rx * ex + ry * ey + rz * ez)
  out[13] = -(sx * ex + sy * ey + sz * ez)
  out[14] = fx * ex + fy * ey + fz * ez
  out[15] = 1
}

export function mat4Multiply(out: Float32Array, a: Float32Array, b: Float32Array) {
  const a0 = a[0], a1 = a[1], a2 = a[2], a3 = a[3]
  const a4 = a[4], a5 = a[5], a6 = a[6], a7 = a[7]
  const a8 = a[8], a9 = a[9], a10 = a[10], a11 = a[11]
  const a12 = a[12], a13 = a[13], a14 = a[14], a15 = a[15]

  const b0 = b[0], b1 = b[1], b2 = b[2], b3 = b[3]
  const b4 = b[4], b5 = b[5], b6 = b[6], b7 = b[7]
  const b8 = b[8], b9 = b[9], b10 = b[10], b11 = b[11]
  const b12 = b[12], b13 = b[13], b14 = b[14], b15 = b[15]

  out[0]  = a0 * b0 + a4 * b1 + a8 * b2 + a12 * b3
  out[1]  = a1 * b0 + a5 * b1 + a9 * b2 + a13 * b3
  out[2]  = a2 * b0 + a6 * b1 + a10 * b2 + a14 * b3
  out[3]  = a3 * b0 + a7 * b1 + a11 * b2 + a15 * b3
  out[4]  = a0 * b4 + a4 * b5 + a8 * b6 + a12 * b7
  out[5]  = a1 * b4 + a5 * b5 + a9 * b6 + a13 * b7
  out[6]  = a2 * b4 + a6 * b5 + a10 * b6 + a14 * b7
  out[7]  = a3 * b4 + a7 * b5 + a11 * b6 + a15 * b7
  out[8]  = a0 * b8 + a4 * b9 + a8 * b10 + a12 * b11
  out[9]  = a1 * b8 + a5 * b9 + a9 * b10 + a13 * b11
  out[10] = a2 * b8 + a6 * b9 + a10 * b10 + a14 * b11
  out[11] = a3 * b8 + a7 * b9 + a11 * b10 + a15 * b11
  out[12] = a0 * b12 + a4 * b13 + a8 * b14 + a12 * b15
  out[13] = a1 * b12 + a5 * b13 + a9 * b14 + a13 * b15
  out[14] = a2 * b12 + a6 * b13 + a10 * b14 + a14 * b15
  out[15] = a3 * b12 + a7 * b13 + a11 * b14 + a15 * b15
}

// WebGPU orthographic projection (Z maps to [0, 1])
export function mat4Ortho(
  out: Float32Array,
  left: number,
  right: number,
  bottom: number,
  top: number,
  near: number,
  far: number,
) {
  const lr = 1 / (left - right)
  const bt = 1 / (bottom - top)
  const nf = 1 / (near - far)
  out[0] = -2 * lr
  out[1] = 0
  out[2] = 0
  out[3] = 0
  out[4] = 0
  out[5] = -2 * bt
  out[6] = 0
  out[7] = 0
  out[8] = 0
  out[9] = 0
  out[10] = nf
  out[11] = 0
  out[12] = (left + right) * lr
  out[13] = (top + bottom) * bt
  out[14] = near * nf
  out[15] = 1
}

export function mat4FromEulerXYZ(out: Float32Array, rx: number, ry: number, rz: number) {
  const cx = Math.cos(rx),
    sx = Math.sin(rx)
  const cy = Math.cos(ry),
    sy = Math.sin(ry)
  const cz = Math.cos(rz),
    sz = Math.sin(rz)
  out[0] = cz * cy
  out[1] = sz * cy
  out[2] = -sy
  out[3] = 0
  out[4] = cz * sy * sx - sz * cx
  out[5] = sz * sy * sx + cz * cx
  out[6] = cy * sx
  out[7] = 0
  out[8] = cz * sy * cx + sz * sx
  out[9] = sz * sy * cx - cz * sx
  out[10] = cy * cx
  out[11] = 0
  out[12] = 0
  out[13] = 0
  out[14] = 0
  out[15] = 1
}

/** Compose a TRS (translate, rotate XYZ euler, scale) matrix directly into out. */
export function mat4ComposeTRS(
  out: Float32Array,
  px: number,
  py: number,
  pz: number,
  rx: number,
  ry: number,
  rz: number,
  sx: number,
  sy: number,
  sz: number,
) {
  const cx = Math.cos(rx),
    snx = Math.sin(rx)
  const cy = Math.cos(ry),
    sy2 = Math.sin(ry)
  const cz = Math.cos(rz),
    snz = Math.sin(rz)
  out[0] = cz * cy * sx
  out[1] = snz * cy * sx
  out[2] = -sy2 * sx
  out[3] = 0
  out[4] = (cz * sy2 * snx - snz * cx) * sy
  out[5] = (snz * sy2 * snx + cz * cx) * sy
  out[6] = cy * snx * sy
  out[7] = 0
  out[8] = (cz * sy2 * cx + snz * snx) * sz
  out[9] = (snz * sy2 * cx - cz * snx) * sz
  out[10] = cy * cx * sz
  out[11] = 0
  out[12] = px
  out[13] = py
  out[14] = pz
  out[15] = 1
}

/** Compose a TQS (translate, quaternion, scale) matrix directly into out. */
export function mat4ComposeTQS(
  out: Float32Array,
  px: number,
  py: number,
  pz: number,
  qx: number,
  qy: number,
  qz: number,
  qw: number,
  sx: number,
  sy: number,
  sz: number,
) {
  const x2 = qx + qx,
    y2 = qy + qy,
    z2 = qz + qz
  const xx = qx * x2,
    xy = qx * y2,
    xz = qx * z2
  const yy = qy * y2,
    yz = qy * z2,
    zz = qz * z2
  const wx = qw * x2,
    wy = qw * y2,
    wz = qw * z2

  out[0] = (1 - (yy + zz)) * sx
  out[1] = (xy + wz) * sx
  out[2] = (xz - wy) * sx
  out[3] = 0
  out[4] = (xy - wz) * sy
  out[5] = (1 - (xx + zz)) * sy
  out[6] = (yz + wx) * sy
  out[7] = 0
  out[8] = (xz + wy) * sz
  out[9] = (yz - wx) * sz
  out[10] = (1 - (xx + yy)) * sz
  out[11] = 0
  out[12] = px
  out[13] = py
  out[14] = pz
  out[15] = 1
}

/** Invert a 4×4 matrix. Returns false if singular. */
export function mat4Invert(out: Float32Array, m: Float32Array): boolean {
  const m00 = m[0],
    m01 = m[1],
    m02 = m[2],
    m03 = m[3]
  const m10 = m[4],
    m11 = m[5],
    m12 = m[6],
    m13 = m[7]
  const m20 = m[8],
    m21 = m[9],
    m22 = m[10],
    m23 = m[11]
  const m30 = m[12],
    m31 = m[13],
    m32 = m[14],
    m33 = m[15]
  const b00 = m00 * m11 - m01 * m10,
    b01 = m00 * m12 - m02 * m10
  const b02 = m00 * m13 - m03 * m10,
    b03 = m01 * m12 - m02 * m11
  const b04 = m01 * m13 - m03 * m11,
    b05 = m02 * m13 - m03 * m12
  const b06 = m20 * m31 - m21 * m30,
    b07 = m20 * m32 - m22 * m30
  const b08 = m20 * m33 - m23 * m30,
    b09 = m21 * m32 - m22 * m31
  const b10 = m21 * m33 - m23 * m31,
    b11 = m22 * m33 - m23 * m32
  let det = b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06
  if (Math.abs(det) < 1e-10) return false
  det = 1 / det
  out[0] = (m11 * b11 - m12 * b10 + m13 * b09) * det
  out[1] = (m02 * b10 - m01 * b11 - m03 * b09) * det
  out[2] = (m31 * b05 - m32 * b04 + m33 * b03) * det
  out[3] = (m22 * b04 - m21 * b05 - m23 * b03) * det
  out[4] = (m12 * b08 - m10 * b11 - m13 * b07) * det
  out[5] = (m00 * b11 - m02 * b08 + m03 * b07) * det
  out[6] = (m32 * b02 - m30 * b05 - m33 * b01) * det
  out[7] = (m20 * b05 - m22 * b02 + m23 * b01) * det
  out[8] = (m10 * b10 - m11 * b08 + m13 * b06) * det
  out[9] = (m01 * b08 - m00 * b10 - m03 * b06) * det
  out[10] = (m30 * b04 - m31 * b02 + m33 * b00) * det
  out[11] = (m21 * b02 - m20 * b04 - m23 * b00) * det
  out[12] = (m11 * b07 - m10 * b09 - m12 * b06) * det
  out[13] = (m00 * b09 - m01 * b07 + m02 * b06) * det
  out[14] = (m31 * b01 - m30 * b03 - m32 * b00) * det
  out[15] = (m20 * b03 - m21 * b01 + m22 * b00) * det
  return true
}

/** Set out to identity. */
export function mat4Identity(out: Float32Array) {
  out[0] = 1
  out[1] = 0
  out[2] = 0
  out[3] = 0
  out[4] = 0
  out[5] = 1
  out[6] = 0
  out[7] = 0
  out[8] = 0
  out[9] = 0
  out[10] = 1
  out[11] = 0
  out[12] = 0
  out[13] = 0
  out[14] = 0
  out[15] = 1
}
