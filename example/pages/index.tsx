import { useCallback, useEffect, useRef, useState } from 'react'

import {
  ACESFilmicToneMapping,
  AgXToneMapping,
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
  NeutralToneMapping,
  NoToneMapping,
  OrbitControls,
  PerspectiveCamera,
  PlaneGeometry,
  Scene,
  SoftToneMapping,
  SphereGeometry,
  TetrahedronGeometry,
  TorusGeometry,
  WebGPURenderer,
} from 'nanothree'

import type { Shading, ToneMapping } from 'nanothree'
import type { BufferGeometry } from 'nanothree'

// ─── Geometry generators with random variations ─────────────────────

function randomRange(min: number, max: number) {
  return min + Math.random() * (max - min)
}

function randomInt(min: number, max: number) {
  return Math.floor(randomRange(min, max + 1))
}

type Complexity = 'low' | 'high'

function makeRandomGeometry(complexity: Complexity = 'low'): BufferGeometry {
  const hi = complexity === 'high'
  const type = randomInt(0, 8)
  switch (type) {
    case 0:
      return new BoxGeometry(
        randomRange(0.2, 1.5),
        randomRange(0.2, 2),
        randomRange(0.2, 1.5),
        hi ? randomInt(8, 16) : randomInt(1, 3),
        hi ? randomInt(8, 16) : randomInt(1, 3),
        hi ? randomInt(8, 16) : randomInt(1, 3),
      )
    case 1:
      return new SphereGeometry(
        randomRange(0.2, 0.8),
        hi ? randomInt(32, 64) : randomInt(6, 24),
        hi ? randomInt(16, 32) : randomInt(4, 16),
      )
    case 2:
      return new CapsuleGeometry(
        randomRange(0.1, 0.5),
        randomRange(0.2, 1.2),
        hi ? randomInt(8, 16) : randomInt(2, 8),
        hi ? randomInt(16, 32) : randomInt(4, 16),
      )
    case 3:
      return new CylinderGeometry(
        randomRange(0.1, 0.6),
        randomRange(0.1, 0.8),
        randomRange(0.3, 2),
        hi ? randomInt(32, 64) : randomInt(4, 24),
        hi ? randomInt(4, 8) : randomInt(1, 4),
      )
    case 4:
      return new ConeGeometry(
        randomRange(0.2, 0.8),
        randomRange(0.4, 2),
        hi ? randomInt(32, 64) : randomInt(4, 24),
        hi ? randomInt(4, 8) : randomInt(1, 3),
      )
    case 5:
      return new CircleGeometry(randomRange(0.2, 0.8), hi ? randomInt(32, 64) : randomInt(4, 24))
    case 6:
      return new TorusGeometry(
        randomRange(0.3, 0.7),
        randomRange(0.05, 0.25),
        hi ? randomInt(16, 32) : randomInt(4, 16),
        hi ? randomInt(32, 64) : randomInt(8, 32),
      )
    case 7:
      return new TetrahedronGeometry(randomRange(0.3, 0.8))
    default:
      return new PlaneGeometry(
        randomRange(0.3, 1.5),
        randomRange(0.3, 1.5),
        hi ? randomInt(8, 16) : randomInt(1, 4),
        hi ? randomInt(8, 16) : randomInt(1, 4),
      )
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

// Vivid, fully-saturated hue for emissive — picks primary/secondary-ish
// colors so the bloom halo reads as a clear glow rather than tinted white.
function makeEmissiveColor(): Color {
  const h = Math.random()
  const a = 0.5
  const f = (n: number) => {
    const k = (n + h * 12) % 12
    return 0.5 - a * Math.max(Math.min(k - 3, 9 - k, 1), -1)
  }
  return new Color(f(0), f(8), f(4))
}

// ─── Demos ──────────────────────────────────────────────────────────

type Demo = 'static' | 'skinned'

const STATIC_CASES = [
  { label: '1,000 objects', count: 1000, complexity: 'low' },
  { label: '5,000 objects', count: 5000, complexity: 'low' },
  { label: '10,000 objects', count: 10000, complexity: 'low' },
  { label: '20,000 objects', count: 20000, complexity: 'low' },
  { label: '1,000 high-poly', count: 1000, complexity: 'high' },
  { label: '5,000 high-poly', count: 5000, complexity: 'high' },
  { label: '10,000 high-poly', count: 10000, complexity: 'high' },
] as const satisfies ReadonlyArray<{ label: string; count: number; complexity: Complexity }>

const SKINNED_COUNTS = [100, 200, 500, 1000] as const

const TONEMAP_OPTIONS: ReadonlyArray<{ label: string; value: ToneMapping }> = [
  { label: 'None', value: NoToneMapping },
  { label: 'Soft', value: SoftToneMapping },
  { label: 'Neutral', value: NeutralToneMapping },
  { label: 'AgX', value: AgXToneMapping },
  { label: 'ACES', value: ACESFilmicToneMapping },
]

// ─── Page ───────────────────────────────────────────────────────────

const IndexPage = () => {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [demo, setDemo] = useState<Demo>('static')
  const [staticCaseIndex, setStaticCaseIndex] = useState(0)
  const [skinnedCount, setSkinnedCount] = useState<number>(100)
  const [shadows, setShadows] = useState(false)
  const shadowsRef = useRef(false)
  shadowsRef.current = shadows
  const [bloom, setBloom] = useState(false)
  const bloomRef = useRef(false)
  bloomRef.current = bloom
  const [toneMapping, setToneMapping] = useState<ToneMapping>(NoToneMapping)
  const toneMappingRef = useRef<ToneMapping>(NoToneMapping)
  toneMappingRef.current = toneMapping
  // Half-Lambert on by default — softer wraparound gives more color dynamics
  // than plain Lambert across the whole [0,1] envelope.
  const [halfLambert, setHalfLambert] = useState(true)
  const halfLambertRef = useRef(true)
  halfLambertRef.current = halfLambert
  // Light split: 0 = fully ambient (flat), 1 = fully directional (max shading).
  // ambient.intensity + dirLight.intensity always == 1 so the brightest lit
  // pixel hits exactly 1.0 regardless of where the slider sits.
  const [lightRatio, setLightRatio] = useState(0.7)
  const lightRatioRef = useRef(0.7)
  lightRatioRef.current = lightRatio
  const [fps, setFps] = useState(0)
  const [drawCalls, setDrawCalls] = useState(0)
  const [triangles, setTriangles] = useState(0)
  const cleanupRef = useRef<(() => void) | null>(null)

  const runStatic = useCallback((canvas: HTMLCanvasElement, count: number, complexity: Complexity) => {
    const renderer = new WebGPURenderer({ canvas })
    const scene = new Scene()
    const camera = new PerspectiveCamera(60, canvas.clientWidth / canvas.clientHeight, 0.1, 500)

    // White ambient + white directional, intensities sum to 1 so the brightest
    // lit pixel hits exactly 1.0 (both Lambert and half-Lambert peak at the
    // same value). The split is driven from the Light Ratio slider per frame.
    const ambient = new AmbientLight(0xffffff, 1 - lightRatioRef.current)
    scene.add(ambient)

    // Spread radius scales with count
    const spread = Math.cbrt(count) * 1.5

    const dirLight = new DirectionalLight(0xffffff, lightRatioRef.current)
    dirLight.position.set(spread, spread * 2, spread * 1.5)
    dirLight.shadow.camera.left = -spread * 1.5
    dirLight.shadow.camera.right = spread * 1.5
    dirLight.shadow.camera.top = spread * 1.5
    dirLight.shadow.camera.bottom = -spread * 1.5
    dirLight.shadow.camera.near = 0.5
    dirLight.shadow.camera.far = spread * 6
    scene.add(dirLight)
    const meshes: Mesh[] = []
    // Materials with an emissive color set. Their `emissiveIntensity` is
    // flipped between 0 and 1 in lockstep with the Bloom toggle so that
    // when bloom is off the scene shows pure Lambert color, no emissive.
    const emissiveMats: MeshLambertMaterial[] = []

    const initialShading: Shading = halfLambertRef.current ? 'half-lambert' : 'lambert'
    // All Lambert materials created here — flipped en masse on toggle.
    const lambertMats: MeshLambertMaterial[] = []

    for (let i = 0; i < count; i++) {
      const geo = makeRandomGeometry(complexity)
      // 20% of meshes get a vivid emissive so the Bloom toggle has obvious
      // glowing sources scattered through the scene.
      const isEmissive = Math.random() < 0.2
      const mat = new MeshLambertMaterial({
        color: makeRandomColor(),
        emissive: isEmissive ? makeEmissiveColor() : undefined,
        emissiveIntensity: 0,
        shading: initialShading,
      })
      lambertMats.push(mat)
      if (isEmissive) emissiveMats.push(mat)
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
    let lastBloom: boolean | null = null
    let lastHalfLambert: boolean | null = null

    const animate = async () => {
      if (!inited) {
        await renderer.init()
        // Lit Lambert pixels peak at exactly 1.0 thanks to the normalized
        // ambient + directional split. Threshold 1.01 keeps lit surfaces
        // sub-threshold so only emissive (which sits in HDR space at 4+) blooms.
        renderer.bloom.threshold = 1.01
        renderer.bloom.strength = 0.8
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
      // Light ratio: ambient = 1 - r, directional = r. Sum always 1.
      const r = lightRatioRef.current
      ambient.intensity = 1 - r
      dirLight.intensity = r

      const hl = halfLambertRef.current
      if (hl !== lastHalfLambert) {
        const mode: Shading = hl ? 'half-lambert' : 'lambert'
        for (const m of lambertMats) m.shading = mode
        lastHalfLambert = hl
      }

      const bloomOn = bloomRef.current
      renderer.bloom.enabled = bloomOn
      renderer.bloom.toneMapping = toneMappingRef.current
      if (bloomOn !== lastBloom) {
        // In HDR, intensity 4 pushes emissive well above the bloom threshold
        // so emissive surfaces clearly bloom while lit Lambert stays clean.
        const intensity = bloomOn ? 4 : 0
        for (const m of emissiveMats) m.emissiveIntensity = intensity
        lastBloom = bloomOn
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

    // White ambient + white directional, intensities sum to 1.
    const ambient = new AmbientLight(0xffffff, 1 - lightRatioRef.current)
    scene.add(ambient)
    const mixers: AnimationMixer[] = []
    const spread = Math.sqrt(count) * 2
    const skinnedMeshes: import('nanothree').Object3D[] = []

    const dirLight = new DirectionalLight(0xffffff, lightRatioRef.current)
    dirLight.position.set(spread * 0.5, spread, spread * 0.7)
    dirLight.shadow.mapSize.set(2048, 2048)
    dirLight.shadow.camera.left = -spread
    dirLight.shadow.camera.right = spread
    dirLight.shadow.camera.top = spread
    dirLight.shadow.camera.bottom = -spread
    dirLight.shadow.camera.near = 0.5
    dirLight.shadow.camera.far = spread * 4
    scene.add(dirLight)

    const initialShading: Shading = halfLambertRef.current ? 'half-lambert' : 'lambert'
    // All Lambert materials we control. Walked en masse on the shading toggle.
    const lambertMats: MeshLambertMaterial[] = []

    // Ground
    const groundMat = new MeshLambertMaterial({ color: 0x445544, shading: initialShading })
    lambertMats.push(groundMat)
    const ground = new Mesh(new PlaneGeometry(spread * 4, spread * 4), groundMat)
    ground.rotation.x = -Math.PI / 2
    scene.add(ground)

    // Collected SkinnedMesh materials whose `emissive` was tinted on load.
    // Intensity is flipped between 0 and 1 in sync with the Bloom toggle so
    // that when bloom is off the characters render with no emissive at all.
    const emissiveMats: MeshLambertMaterial[] = []

    const applySkinnedEmissive = (
      node: { children: unknown[]; isSkinnedMesh?: boolean; material?: MeshLambertMaterial },
      emissive: Color,
    ) => {
      if (node.isSkinnedMesh && node.material) {
        node.material.emissive = emissive
        node.material.emissiveIntensity = bloomRef.current ? 4 : 0
        node.material.shading = halfLambertRef.current ? 'half-lambert' : 'lambert'
        emissiveMats.push(node.material)
        lambertMats.push(node.material)
      }
      for (const child of node.children) applySkinnedEmissive(child as typeof node, emissive)
    }

    // GLTFLoader caches the first load and deep-clones on each subsequent call
    const loader = new GLTFLoader()
    for (let i = 0; i < count; i++) {
      loader.load(
        '/michelle.glb',
        result => {
          result.scene.position.set((Math.random() - 0.5) * spread, 0, (Math.random() - 0.5) * spread)
          result.scene.rotation.set(0, Math.random() * Math.PI * 2, 0)
          // Pick a per-character hue so neighbouring characters glow different colors.
          applySkinnedEmissive(
            result.scene as unknown as Parameters<typeof applySkinnedEmissive>[0],
            makeEmissiveColor(),
          )
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
    let lastBloom: boolean | null = null
    let lastHalfLambert: boolean | null = null

    const animate = async () => {
      if (!inited) {
        await renderer.init()
        // Lit pixels peak at exactly 1.0 from the normalized light split.
        // Threshold 1.01 isolates emissive glow (which lives in HDR > 1).
        renderer.bloom.threshold = 1.01
        renderer.bloom.strength = 0.8
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

      // Light ratio: ambient = 1 - r, directional = r. Sum always 1.
      const r = lightRatioRef.current
      ambient.intensity = 1 - r
      dirLight.intensity = r

      const hl = halfLambertRef.current
      if (hl !== lastHalfLambert) {
        const mode: Shading = hl ? 'half-lambert' : 'lambert'
        for (const m of lambertMats) m.shading = mode
        lastHalfLambert = hl
      }

      const bloomOn = bloomRef.current
      renderer.bloom.enabled = bloomOn
      renderer.bloom.toneMapping = toneMappingRef.current
      if (bloomOn !== lastBloom) {
        // In HDR, intensity 4 pushes emissive well above the bloom threshold
        // so emissive surfaces clearly bloom while lit Lambert stays clean.
        const intensity = bloomOn ? 4 : 0
        for (const m of emissiveMats) m.emissiveIntensity = intensity
        lastBloom = bloomOn
      }
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
      const c = STATIC_CASES[staticCaseIndex]!
      cleanupRef.current = runStatic(canvas, c.count, c.complexity)
    } else {
      cleanupRef.current = runSkinned(canvas, skinnedCount)
    }
    return () => {
      cleanupRef.current?.()
      cleanupRef.current = null
    }
  }, [demo, staticCaseIndex, skinnedCount, runStatic, runSkinned])

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
            value={staticCaseIndex}
            onChange={e => setStaticCaseIndex(Number(e.target.value))}
            className="cursor-pointer rounded bg-white/10 px-3 py-1.5 text-white/80 hover:bg-white/20"
          >
            {STATIC_CASES.map((c, i) => (
              <option key={c.label} value={i}>
                {c.label}
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
        <button
          onClick={() => setBloom(b => !b)}
          className={`cursor-pointer rounded px-3 py-1.5 ${bloom ? 'bg-fuchsia-500 text-black' : 'bg-white/10 text-white/80 hover:bg-white/20'}`}
        >
          Bloom {bloom ? 'ON' : 'OFF'}
        </button>
        <button
          onClick={() => setHalfLambert(h => !h)}
          className={`cursor-pointer rounded px-3 py-1.5 ${halfLambert ? 'bg-emerald-500 text-black' : 'bg-white/10 text-white/80 hover:bg-white/20'}`}
        >
          Half-Lambert {halfLambert ? 'ON' : 'OFF'}
        </button>
        <label className="flex items-center gap-2 rounded bg-white/10 px-3 py-1.5 text-white/80">
          <span>Light {lightRatio.toFixed(2)}</span>
          <input
            type="range"
            min={0}
            max={1}
            step={0.01}
            value={lightRatio}
            onChange={e => setLightRatio(Number(e.target.value))}
            className="cursor-pointer"
          />
        </label>
        <select
          value={toneMapping}
          onChange={e => setToneMapping(Number(e.target.value) as ToneMapping)}
          className="cursor-pointer rounded bg-white/10 px-3 py-1.5 text-white/80 hover:bg-white/20"
        >
          {TONEMAP_OPTIONS.map(o => (
            <option key={o.value} value={o.value}>
              Tonemap: {o.label}
            </option>
          ))}
        </select>
      </div>
    </div>
  )
}

export default IndexPage
