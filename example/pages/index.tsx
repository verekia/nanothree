import { useCallback, useEffect, useRef, useState } from 'react'

import {
  AmbientLight,
  AnimationMixer,
  BoxGeometry,
  CapsuleGeometry,
  CircleGeometry,
  Color,
  ConeGeometry,
  CylinderGeometry,
  DirectionalLight,
  GLTFLoader,
  Mesh,
  MeshLambertMaterial,
  OrbitControls,
  PerspectiveCamera,
  PlaneGeometry,
  Scene,
  SphereGeometry,
  TetrahedronGeometry,
  TorusGeometry,
  WebGPURenderer,
} from 'nanothree'

import type { BufferGeometry } from 'nanothree'

// ─── Geometry generators with random variations ─────────────────────

function randomRange(min: number, max: number) {
  return min + Math.random() * (max - min)
}

function randomInt(min: number, max: number) {
  return Math.floor(randomRange(min, max + 1))
}

function makeRandomGeometry(): BufferGeometry {
  const type = randomInt(0, 8)
  switch (type) {
    case 0:
      return new BoxGeometry(
        randomRange(0.2, 1.5),
        randomRange(0.2, 2),
        randomRange(0.2, 1.5),
        randomInt(1, 3),
        randomInt(1, 3),
        randomInt(1, 3),
      )
    case 1:
      return new SphereGeometry(randomRange(0.2, 0.8), randomInt(6, 24), randomInt(4, 16))
    case 2:
      return new CapsuleGeometry(randomRange(0.1, 0.5), randomRange(0.2, 1.2), randomInt(2, 8), randomInt(4, 16))
    case 3:
      return new CylinderGeometry(
        randomRange(0.1, 0.6),
        randomRange(0.1, 0.8),
        randomRange(0.3, 2),
        randomInt(4, 24),
        randomInt(1, 4),
      )
    case 4:
      return new ConeGeometry(randomRange(0.2, 0.8), randomRange(0.4, 2), randomInt(4, 24), randomInt(1, 3))
    case 5:
      return new CircleGeometry(randomRange(0.2, 0.8), randomInt(4, 24))
    case 6:
      return new TorusGeometry(randomRange(0.3, 0.7), randomRange(0.05, 0.25), randomInt(4, 16), randomInt(8, 32))
    case 7:
      return new TetrahedronGeometry(randomRange(0.3, 0.8))
    default:
      return new PlaneGeometry(randomRange(0.3, 1.5), randomRange(0.3, 1.5), randomInt(1, 4), randomInt(1, 4))
  }
}

function makeRandomColor(): Color {
  const h = Math.random()
  const s = 0.5 + Math.random() * 0.5
  const l = 0.3 + Math.random() * 0.4
  // HSL to RGB
  const a = s * Math.min(l, 1 - l)
  const f = (n: number) => {
    const k = (n + h * 12) % 12
    return l - a * Math.max(Math.min(k - 3, 9 - k, 1), -1)
  }
  return new Color(f(0), f(8), f(4))
}

// ─── Demos ──────────────────────────────────────────────────────────

type Demo = 'static' | 'skinned'

const STATIC_COUNTS = [1000, 5000, 10000, 20000] as const
const SKINNED_COUNTS = [100, 200, 500, 1000] as const

// ─── Page ───────────────────────────────────────────────────────────

const IndexPage = () => {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [demo, setDemo] = useState<Demo>('static')
  const [staticCount, setStaticCount] = useState<number>(1000)
  const [skinnedCount, setSkinnedCount] = useState<number>(100)
  const [shadows, setShadows] = useState(false)
  const shadowsRef = useRef(false)
  shadowsRef.current = shadows
  const [fps, setFps] = useState(0)
  const [drawCalls, setDrawCalls] = useState(0)
  const [triangles, setTriangles] = useState(0)
  const cleanupRef = useRef<(() => void) | null>(null)

  const runStatic = useCallback((canvas: HTMLCanvasElement, count: number) => {
    const renderer = new WebGPURenderer({ canvas })
    const scene = new Scene()
    const camera = new PerspectiveCamera(60, canvas.clientWidth / canvas.clientHeight, 0.1, 500)

    const ambient = new AmbientLight(0x606080, 0.5)
    scene.add(ambient)

    // Spread radius scales with count
    const spread = Math.cbrt(count) * 1.5

    const dirLight = new DirectionalLight(0xffffff, 1)
    dirLight.position.set(spread, spread * 2, spread * 1.5)
    dirLight.shadow.camera.left = -spread * 1.5
    dirLight.shadow.camera.right = spread * 1.5
    dirLight.shadow.camera.top = spread * 1.5
    dirLight.shadow.camera.bottom = -spread * 1.5
    dirLight.shadow.camera.near = 0.5
    dirLight.shadow.camera.far = spread * 6
    scene.add(dirLight)
    const meshes: Mesh[] = []

    for (let i = 0; i < count; i++) {
      const geo = makeRandomGeometry()
      const mat = new MeshLambertMaterial({ color: makeRandomColor() })
      const mesh = new Mesh(geo, mat)
      mesh.position.set(
        (Math.random() - 0.5) * spread * 2,
        (Math.random() - 0.5) * spread,
        (Math.random() - 0.5) * spread * 2,
      )
      mesh.rotation.set(Math.random() * Math.PI * 2, Math.random() * Math.PI * 2, Math.random() * Math.PI * 2)
      const s = randomRange(0.3, 1.5)
      mesh.scale.set(s, s, s)
      scene.add(mesh)
      meshes.push(mesh)
    }

    // Position camera and create controls
    camera.position.set(spread * 1.2, spread * 0.8, spread * 1.8)
    const orbit = new OrbitControls(camera, canvas)
    orbit.minDistance = 5
    orbit.maxDistance = spread * 5

    let raf = 0
    let lastTime = 0
    let inited = false
    let frameCount = 0
    let fpsAccum = 0

    const animate = async () => {
      if (!inited) {
        await renderer.init()
        inited = true
      }
      raf = requestAnimationFrame(animate)
      const now = performance.now() / 1000
      const dt = lastTime ? now - lastTime : 1 / 60
      lastTime = now
      frameCount++
      fpsAccum += dt
      if (fpsAccum >= 0.5) {
        setFps(Math.round(frameCount / fpsAccum))
        frameCount = 0
        fpsAccum = 0
      }

      // Sync shadows from ref
      const s = shadowsRef.current
      renderer.shadowMap.enabled = s
      dirLight.castShadow = s
      for (const m of meshes) {
        m.rotation.y += dt * 1.5
        m.castShadow = s
        m.receiveShadow = s
      }

      orbit.update()
      renderer.render(scene, camera)
      setDrawCalls(renderer.info.drawCalls)
      setTriangles(renderer.info.triangles)
    }
    animate()

    return () => {
      cancelAnimationFrame(raf)
      orbit.dispose()
    }
  }, [])

  const runSkinned = useCallback((canvas: HTMLCanvasElement, count: number) => {
    const renderer = new WebGPURenderer({ canvas })
    const scene = new Scene()
    const camera = new PerspectiveCamera(60, canvas.clientWidth / canvas.clientHeight, 0.1, 500)

    const ambient = new AmbientLight(0x506070, 0.6)
    scene.add(ambient)
    const mixers: AnimationMixer[] = []
    const spread = Math.sqrt(count) * 2
    const skinnedMeshes: import('nanothree').Object3D[] = []

    const dirLight = new DirectionalLight(0xffffff, 1.2)
    dirLight.position.set(spread * 0.5, spread, spread * 0.7)
    dirLight.shadow.mapSize.set(2048, 2048)
    dirLight.shadow.camera.left = -spread
    dirLight.shadow.camera.right = spread
    dirLight.shadow.camera.top = spread
    dirLight.shadow.camera.bottom = -spread
    dirLight.shadow.camera.near = 0.5
    dirLight.shadow.camera.far = spread * 4
    scene.add(dirLight)

    // Ground
    const ground = new Mesh(new PlaneGeometry(spread * 4, spread * 4), new MeshLambertMaterial({ color: 0x445544 }))
    ground.rotation.x = -Math.PI / 2
    scene.add(ground)

    // GLTFLoader caches the first load and deep-clones on each subsequent call
    const loader = new GLTFLoader()
    for (let i = 0; i < count; i++) {
      loader.load(
        '/michelle.glb',
        result => {
          result.scene.position.set((Math.random() - 0.5) * spread, 0, (Math.random() - 0.5) * spread)
          result.scene.rotation.set(0, Math.random() * Math.PI * 2, 0)
          scene.add(result.scene)
          skinnedMeshes.push(result.scene)

          if (result.animations.length > 0) {
            const mixer = new AnimationMixer(result.scene)
            const action = mixer.clipAction(result.animations[0]!)
            action.play()
            // Offset animation time so they're not all in sync
            action._advance(Math.random() * result.animations[0]!.duration)
            mixers.push(mixer)
          }
        },
        undefined,
        err => console.error('Failed to load michelle.glb:', err),
      )
    }

    camera.position.set(spread * 0.7, spread * 0.5, spread * 1.1)
    const orbit = new OrbitControls(camera, canvas)
    orbit.target.y = 1
    orbit.minDistance = 5
    orbit.maxDistance = spread * 5

    let raf = 0
    let lastTime = 0
    let inited = false
    let frameCount = 0
    let fpsAccum = 0

    const animate = async () => {
      if (!inited) {
        await renderer.init()
        inited = true
      }
      raf = requestAnimationFrame(animate)
      const now = performance.now() / 1000
      const dt = lastTime ? now - lastTime : 1 / 60
      lastTime = now
      frameCount++
      fpsAccum += dt
      if (fpsAccum >= 0.5) {
        setFps(Math.round(frameCount / fpsAccum))
        frameCount = 0
        fpsAccum = 0
      }

      // Sync shadows from ref
      const s = shadowsRef.current
      renderer.shadowMap.enabled = s
      dirLight.castShadow = s
      ground.receiveShadow = s
      for (const m of skinnedMeshes) {
        m.castShadow = s
        m.receiveShadow = s
      }

      for (const mixer of mixers) mixer.update(dt)
      orbit.update()
      renderer.render(scene, camera)
      setDrawCalls(renderer.info.drawCalls)
      setTriangles(renderer.info.triangles)
    }
    animate()

    return () => {
      cancelAnimationFrame(raf)
      orbit.dispose()
    }
  }, [])

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    cleanupRef.current?.()
    if (demo === 'static') {
      cleanupRef.current = runStatic(canvas, staticCount)
    } else {
      cleanupRef.current = runSkinned(canvas, skinnedCount)
    }
    return () => {
      cleanupRef.current?.()
      cleanupRef.current = null
    }
  }, [demo, staticCount, skinnedCount, runStatic, runSkinned])

  return (
    <div className="fixed inset-0 bg-black">
      <canvas ref={canvasRef} className="h-full w-full" />

      {/* Top-left: title + stats */}
      <div className="fixed top-4 left-4 font-mono text-sm text-white/80">
        <h1 className="mb-1 text-lg font-bold">nanothree</h1>
        <a
          href="https://github.com/verekia/nanothree"
          target="_blank"
          rel="noopener noreferrer"
          className="mt-1 inline-flex items-center gap-1.5 text-xs text-white/50 hover:text-white/80"
        >
          <svg viewBox="0 0 16 16" fill="currentColor" className="size-3.5">
            <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z" />
          </svg>
          GitHub
        </a>

        <div className="mt-2 text-xs text-white/60">
          <p>{fps} FPS</p>
          <p>{drawCalls.toLocaleString()} draw calls</p>
          <p>{triangles.toLocaleString()} triangles</p>
        </div>
      </div>

      {/* Top-right: controls */}
      <div className="fixed top-4 right-4 flex items-center gap-3 font-mono text-sm">
        <button
          onClick={() => setDemo('static')}
          className={`cursor-pointer rounded px-3 py-1.5 ${demo === 'static' ? 'bg-white text-black' : 'bg-white/10 text-white/80 hover:bg-white/20'}`}
        >
          Static
        </button>
        <button
          onClick={() => setDemo('skinned')}
          className={`cursor-pointer rounded px-3 py-1.5 ${demo === 'skinned' ? 'bg-white text-black' : 'bg-white/10 text-white/80 hover:bg-white/20'}`}
        >
          Skinned
        </button>
        {demo === 'static' && (
          <select
            value={staticCount}
            onChange={e => setStaticCount(Number(e.target.value))}
            className="cursor-pointer rounded bg-white/10 px-3 py-1.5 text-white/80 hover:bg-white/20"
          >
            {STATIC_COUNTS.map(c => (
              <option key={c} value={c}>
                {c.toLocaleString()} objects
              </option>
            ))}
          </select>
        )}
        {demo === 'skinned' && (
          <select
            value={skinnedCount}
            onChange={e => setSkinnedCount(Number(e.target.value))}
            className="cursor-pointer rounded bg-white/10 px-3 py-1.5 text-white/80 hover:bg-white/20"
          >
            {SKINNED_COUNTS.map(c => (
              <option key={c} value={c}>
                {c.toLocaleString()} characters
              </option>
            ))}
          </select>
        )}
        <button
          onClick={() => setShadows(s => !s)}
          className={`cursor-pointer rounded px-3 py-1.5 ${shadows ? 'bg-amber-500 text-black' : 'bg-white/10 text-white/80 hover:bg-white/20'}`}
        >
          Shadows {shadows ? 'ON' : 'OFF'}
        </button>
      </div>
    </div>
  )
}

export default IndexPage
