import { useEffect, useRef } from 'react'

import {
  AmbientLight,
  AnimationMixer,
  BoxGeometry,
  CapsuleGeometry,
  CircleGeometry,
  ConeGeometry,
  CylinderGeometry,
  DirectionalLight,
  GLTFLoader,
  Mesh,
  MeshLambertMaterial,
  PerspectiveCamera,
  PlaneGeometry,
  Scene,
  SphereGeometry,
  TetrahedronGeometry,
  TorusGeometry,
  WebGPURenderer,
} from 'nanothree'

const IndexPage = () => {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const renderer = new WebGPURenderer({ canvas })
    const scene = new Scene()
    const camera = new PerspectiveCamera(55, canvas.clientWidth / canvas.clientHeight, 0.1, 200)
    camera.position.set(0, 8, 18)
    camera.lookAt(0, 2, 0)

    // Lights
    const ambient = new AmbientLight(0x404060, 0.6)
    scene.add(ambient)

    const dirLight = new DirectionalLight(0xffffff, 1.2)
    dirLight.position.set(5, 10, 7)
    dirLight.castShadow = true
    scene.add(dirLight)

    // Ground plane
    const ground = new Mesh(new PlaneGeometry(30, 30), new MeshLambertMaterial({ color: 0x556655 }))
    ground.rotation.x = -Math.PI / 2
    ground.receiveShadow = true
    scene.add(ground)

    // Primitives arranged in a row
    const primitives: { geometry: ConstructorParameters<typeof Mesh>[0]; x: number; label: string }[] = [
      { geometry: new BoxGeometry(1, 1, 1), x: -8, label: 'Box' },
      { geometry: new SphereGeometry(0.6, 24, 16), x: -6, label: 'Sphere' },
      { geometry: new CapsuleGeometry(0.4, 0.6, 8, 16), x: -4, label: 'Capsule' },
      { geometry: new CylinderGeometry(0.5, 0.5, 1, 24), x: -2, label: 'Cylinder' },
      { geometry: new ConeGeometry(0.6, 1.2, 24), x: 0, label: 'Cone' },
      { geometry: new CircleGeometry(0.6, 24), x: 2, label: 'Circle' },
      { geometry: new TorusGeometry(0.5, 0.2, 12, 32), x: 4, label: 'Torus' },
      { geometry: new TetrahedronGeometry(0.6), x: 6, label: 'Tetra' },
    ]

    const colors = [0xee4444, 0x44bb44, 0x4488ee, 0xeeaa22, 0xcc44cc, 0x44cccc, 0xee8844, 0xaaaa44]
    const meshes: Mesh[] = []

    primitives.forEach(({ geometry, x }, i) => {
      const mat = new MeshLambertMaterial({ color: colors[i]! })
      const mesh = new Mesh(geometry, mat)
      mesh.position.set(x, 1.2, 4)
      mesh.castShadow = true
      mesh.receiveShadow = true
      scene.add(mesh)
      meshes.push(mesh)
    })

    // Michelle model
    let mixer: AnimationMixer | null = null
    const loader = new GLTFLoader()
    loader.load(
      '/michelle.glb',
      result => {
        const model = result.scene
        model.position.set(0, 0, -2)
        model.castShadow = true
        model.receiveShadow = true
        scene.add(model)

        if (result.animations.length > 0) {
          mixer = new AnimationMixer(model)
          const action = mixer.clipAction(result.animations[0]!)
          action.play()
        }
      },
      undefined,
      err => console.error('Failed to load michelle.glb:', err),
    )

    // Orbit controls (manual)
    let rotY = 0
    let rotX = 0.3
    let distance = 18
    let isDragging = false
    let lastX = 0
    let lastY = 0

    const onMouseDown = (e: MouseEvent) => {
      isDragging = true
      lastX = e.clientX
      lastY = e.clientY
    }
    const onMouseMove = (e: MouseEvent) => {
      if (!isDragging) return
      rotY -= (e.clientX - lastX) * 0.005
      rotX -= (e.clientY - lastY) * 0.005
      rotX = Math.max(-Math.PI / 2 + 0.1, Math.min(Math.PI / 2 - 0.1, rotX))
      lastX = e.clientX
      lastY = e.clientY
    }
    const onMouseUp = () => {
      isDragging = false
    }
    const onWheel = (e: WheelEvent) => {
      distance += e.deltaY * 0.01
      distance = Math.max(3, Math.min(50, distance))
    }

    canvas.addEventListener('mousedown', onMouseDown)
    canvas.addEventListener('mousemove', onMouseMove)
    canvas.addEventListener('mouseup', onMouseUp)
    canvas.addEventListener('wheel', onWheel)

    let raf = 0
    let lastTime = 0
    let inited = false

    const animate = async () => {
      if (!inited) {
        await renderer.init()
        inited = true
      }

      raf = requestAnimationFrame(animate)

      const now = performance.now() / 1000
      const dt = lastTime ? now - lastTime : 1 / 60
      lastTime = now

      // Rotate primitives
      meshes.forEach(m => {
        m.rotation.y += dt * 0.5
      })

      // Update animation
      mixer?.update(dt)

      // Update camera from orbit
      const cx = distance * Math.cos(rotX) * Math.sin(rotY)
      const cy = distance * Math.sin(rotX) + 4
      const cz = distance * Math.cos(rotX) * Math.cos(rotY)
      camera.position.set(cx, cy, cz)
      camera.lookAt(0, 2, 0)
      camera.aspect = canvas.clientWidth / canvas.clientHeight

      renderer.render(scene, camera)
    }

    animate()

    return () => {
      cancelAnimationFrame(raf)
      canvas.removeEventListener('mousedown', onMouseDown)
      canvas.removeEventListener('mousemove', onMouseMove)
      canvas.removeEventListener('mouseup', onMouseUp)
      canvas.removeEventListener('wheel', onWheel)
    }
  }, [])

  return (
    <div className="fixed inset-0 bg-black">
      <canvas ref={canvasRef} className="h-full w-full" />
      <div className="fixed top-4 left-4 font-mono text-sm text-white/80">
        <h1 className="mb-1 text-lg font-bold">nanothree</h1>
        <p>Lightweight WebGPU renderer</p>
        <p className="mt-1 text-white/50">Drag to orbit, scroll to zoom</p>
      </div>
    </div>
  )
}

export default IndexPage
