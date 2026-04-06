// OrbitControls for nanothree — matches Three.js OrbitControls behavior.
//
// Left drag: orbit around target
// Right drag: pan (camera-local XY plane)
// Scroll: dolly (zoom in/out)

import type { PerspectiveCamera } from './core'

export class OrbitControls {
  camera: PerspectiveCamera
  domElement: HTMLCanvasElement

  target = { x: 0, y: 0, z: 0 }
  minDistance = 0.1
  maxDistance = Infinity
  enableDamping = false
  dampingFactor = 0.05
  rotateSpeed = 1
  panSpeed = 1
  zoomSpeed = 1

  // Spherical coordinates (theta = horizontal, phi = vertical)
  private _theta = 0
  private _phi = Math.PI / 3
  private _radius: number

  private _dragButton = -1
  private _lastX = 0
  private _lastY = 0

  private _onMouseDown: (e: MouseEvent) => void
  private _onMouseMove: (e: MouseEvent) => void
  private _onMouseUp: (e: MouseEvent) => void
  private _onWheel: (e: WheelEvent) => void
  private _onContextMenu: (e: Event) => void

  constructor(camera: PerspectiveCamera, domElement: HTMLCanvasElement) {
    this.camera = camera
    this.domElement = domElement

    // Compute initial spherical coords from camera position relative to target
    const dx = camera.position.x - this.target.x
    const dy = camera.position.y - this.target.y
    const dz = camera.position.z - this.target.z
    this._radius = Math.sqrt(dx * dx + dy * dy + dz * dz)
    if (this._radius > 1e-6) {
      this._phi = Math.acos(Math.max(-1, Math.min(1, dy / this._radius)))
      this._theta = Math.atan2(dx, dz)
    }

    this._onMouseDown = (e: MouseEvent) => {
      this._dragButton = e.button
      this._lastX = e.clientX
      this._lastY = e.clientY
    }

    this._onMouseMove = (e: MouseEvent) => {
      if (this._dragButton < 0) return
      const dx = e.clientX - this._lastX
      const dy = e.clientY - this._lastY
      this._lastX = e.clientX
      this._lastY = e.clientY

      if (this._dragButton === 0) {
        this._handleRotate(dx, dy)
      } else if (this._dragButton === 2) {
        this._handlePan(dx, dy)
      }
    }

    this._onMouseUp = () => {
      this._dragButton = -1
    }

    this._onWheel = (e: WheelEvent) => {
      this._handleZoom(e.deltaY)
    }

    this._onContextMenu = (e: Event) => {
      e.preventDefault()
    }

    domElement.addEventListener('mousedown', this._onMouseDown)
    domElement.addEventListener('mousemove', this._onMouseMove)
    domElement.addEventListener('mouseup', this._onMouseUp)
    domElement.addEventListener('wheel', this._onWheel)
    domElement.addEventListener('contextmenu', this._onContextMenu)
  }

  private _handleRotate(dx: number, dy: number) {
    // Horizontal: theta (around Y axis)
    this._theta -= ((2 * Math.PI * dx) / this.domElement.clientHeight) * this.rotateSpeed
    // Vertical: phi (from top pole down), mouse up = look up = decrease phi
    this._phi -= ((2 * Math.PI * dy) / this.domElement.clientHeight) * this.rotateSpeed
    // Clamp phi to avoid flipping (slightly above 0 and below PI)
    this._phi = Math.max(0.01, Math.min(Math.PI - 0.01, this._phi))
  }

  private _handlePan(dx: number, dy: number) {
    // Pan in camera-local XY plane, matching Three.js behavior
    const offset = this._radius * Math.tan(((this.camera.fov / 2) * Math.PI) / 180)
    const panX = ((2 * dx * offset) / this.domElement.clientHeight) * this.panSpeed
    const panY = ((2 * dy * offset) / this.domElement.clientHeight) * this.panSpeed

    // Camera right vector from view matrix (world matrix column 0)
    const wm = this.camera._worldMatrix
    const rx = wm[0],
      ry = wm[1],
      rz = wm[2]
    // Camera up vector (world matrix column 1)
    const ux = wm[4],
      uy = wm[5],
      uz = wm[6]

    // Move target: right * -panX + up * panY
    this.target.x += -rx * panX + ux * panY
    this.target.y += -ry * panX + uy * panY
    this.target.z += -rz * panX + uz * panY
  }

  private _handleZoom(deltaY: number) {
    const factor = 1 + deltaY * 0.001 * this.zoomSpeed
    this._radius = Math.max(this.minDistance, Math.min(this.maxDistance, this._radius * factor))
  }

  /** Call each frame to apply orbit state to camera. */
  update() {
    // Spherical to cartesian
    const sinPhi = Math.sin(this._phi)
    const cosPhi = Math.cos(this._phi)
    const sinTheta = Math.sin(this._theta)
    const cosTheta = Math.cos(this._theta)

    this.camera.position.set(
      this.target.x + this._radius * sinPhi * sinTheta,
      this.target.y + this._radius * cosPhi,
      this.target.z + this._radius * sinPhi * cosTheta,
    )
    this.camera.lookAt(this.target.x, this.target.y, this.target.z)
    this.camera.aspect = this.domElement.clientWidth / this.domElement.clientHeight
  }

  dispose() {
    this.domElement.removeEventListener('mousedown', this._onMouseDown)
    this.domElement.removeEventListener('mousemove', this._onMouseMove)
    this.domElement.removeEventListener('mouseup', this._onMouseUp)
    this.domElement.removeEventListener('wheel', this._onWheel)
    this.domElement.removeEventListener('contextmenu', this._onContextMenu)
  }
}
