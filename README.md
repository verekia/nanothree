# nanothree

Lightweight **WebGPU-only** 3D renderer that implements a subset of the [Three.js](https://threejs.org/) API, focusing on performance and small bundle size.

nanothree is not a drop-in replacement for Three.js. It targets the same scene graph patterns and naming conventions, but ships a leaner renderer with fewer abstractions. If you need PBR materials, point lights, morph targets, or post-processing, use Three.js. If you want a fast, minimal WebGPU renderer with a familiar API, nanothree may be a good fit.

⚠️ Disclaimer: This project is entirely vibe coded (as in I didn't look at the code) and it is very likely to go unmaintained. Do not use it for anything important.

## Features

- **WebGPU-native** renderer (no WebGL fallback) with shadow mapping and frustum culling
- **Three.js-compatible scene graph**: `Object3D`, `Group`, `Scene`, `PerspectiveCamera`, `Mesh`, `Line`, `Sprite`
- **9 geometry types**: Box, Sphere, Capsule, Cylinder, Cone, Circle, Plane, Torus, Tetrahedron
- **Lambert and unlit materials** with texture mapping, per-vertex colors, and face culling
- **Custom shaders in WGSL** (not TSL) via `ShaderMaterial`
- **GLTF/GLB loading** with Draco mesh compression and KTX2/Basis texture support (decoders bundled, zero config)
- **Skeletal animation**: `Bone`, `Skeleton`, `SkinnedMesh`, `AnimationMixer`, `AnimationClip`
- **Instancing**: `InstancedMesh` and `InstancedSprite` for rendering thousands of objects
- **Raycasting**: Moller-Trumbore ray-triangle intersection with screen-space picking
- **Shadow mapping**: Directional light depth pass with 4-tap PCF filtering
- **Automatic frustum culling** via bounding spheres
- **Interactive transform gizmo** (translate, rotate, scale with snap-to-grid)
- **Debug helpers**: `CameraHelper`, `DirectionalLightHelper`

## Install

```sh
npm install nanothree
```

## Quick Start

```typescript
import {
  AmbientLight,
  BoxGeometry,
  DirectionalLight,
  Mesh,
  MeshLambertMaterial,
  PerspectiveCamera,
  Scene,
  WebGPURenderer,
} from 'nanothree'

const canvas = document.querySelector('canvas')!
const renderer = new WebGPURenderer({ canvas })
await renderer.init()

const scene = new Scene()
const camera = new PerspectiveCamera(60, canvas.clientWidth / canvas.clientHeight, 0.1, 100)
camera.position.set(0, 2, 5)
camera.lookAt(0, 0, 0)

scene.add(new AmbientLight(0x404040, 0.5))
const light = new DirectionalLight(0xffffff, 1)
light.position.set(5, 10, 5)
scene.add(light)

const cube = new Mesh(new BoxGeometry(1, 1, 1), new MeshLambertMaterial({ color: 0x4488ee }))
scene.add(cube)

function animate() {
  requestAnimationFrame(animate)
  cube.rotation.y += 0.01
  renderer.render(scene, camera)
}
animate()
```

## API Overview

### Renderer

```typescript
const renderer = new WebGPURenderer({ canvas, antialias?: boolean })
await renderer.init()
renderer.render(scene, camera)
renderer.setSize(width, height)
renderer.setPixelRatio(devicePixelRatio)
renderer.shadowMap.enabled = true
renderer.info // { drawCalls, triangles }
```

### Scene Graph

All scene objects extend `Object3D`:

```typescript
object.position.set(x, y, z) // Vector3
object.rotation.set(x, y, z) // Euler (radians)
object.scale.set(x, y, z) // Vector3
object.visible = true
object.castShadow = true
object.receiveShadow = true
object.add(child)
object.remove(child)
```

`PerspectiveCamera` adds a `lookAt` method and supports an orthographic override:

```typescript
const camera = new PerspectiveCamera(fov, aspect, near, far)
camera.lookAt(x, y, z)
camera.orthoOverride = { left, right, bottom, top } // or null for perspective
```

### Geometry

All geometries extend `BufferGeometry` and generate positions, normals, UVs, and indices:

| Class                 | Constructor                                                            |
| --------------------- | ---------------------------------------------------------------------- |
| `BoxGeometry`         | `(width, height, depth, wSegs?, hSegs?, dSegs?)`                       |
| `SphereGeometry`      | `(radius, wSegs?, hSegs?, phiStart?, phiLen?, thetaStart?, thetaLen?)` |
| `CapsuleGeometry`     | `(radius, height, capSegs?, radialSegs?)`                              |
| `CylinderGeometry`    | `(radiusTop, radiusBot, height, radialSegs?, hSegs?, openEnded?)`      |
| `ConeGeometry`        | `(radius, height, radialSegs?, hSegs?, openEnded?)`                    |
| `CircleGeometry`      | `(radius, segments?)`                                                  |
| `PlaneGeometry`       | `(width, height, wSegs?, hSegs?)`                                      |
| `TorusGeometry`       | `(radius, tube, radialSegs?, tubularSegs?, arc?)`                      |
| `TetrahedronGeometry` | `(radius)`                                                             |

Custom geometry via `BufferGeometry`:

```typescript
const geo = new BufferGeometry()
geo.setAttribute('position', new Float32BufferAttribute([...], 3))
geo.setAttribute('normal', new Float32BufferAttribute([...], 3))
geo.setAttribute('uv', new Float32BufferAttribute([...], 2))
geo.setIndex([0, 1, 2, ...])
```

### Materials

**`MeshLambertMaterial`** - Lit material with Lambert shading and shadow support:

```typescript
new MeshLambertMaterial({
  color: 0xff0000, // hex or Color instance
  map: texture, // NanoTexture (albedo)
  wireframe: false,
  side: FrontSide, // FrontSide | BackSide | DoubleSide
  vertexColors: false,
})
```

**`MeshBasicMaterial`** - Unlit, flat color:

```typescript
new MeshBasicMaterial({ color: 0xff0000, wireframe: false, side: FrontSide })
```

**`LineBasicMaterial`** - For `Line` objects:

```typescript
new LineBasicMaterial({ color: 0xffffff })
```

**`SpriteMaterial`** - For camera-facing billboard `Sprite` objects:

```typescript
new SpriteMaterial({ color: 0xffffff, opacity: 1, blending: NormalBlending })
// blending: NormalBlending (alpha) or AdditiveBlending
```

### Custom Shaders (WGSL)

nanothree uses **WGSL** for custom shaders (Three.js uses TSL/GLSL). The renderer auto-prepends a preamble with scene uniforms, so your shader has access to the view-projection matrix, light data, shadow map, and per-object transform:

```typescript
const material = new ShaderMaterial({
  code: /* wgsl */ `
    // Available from preamble:
    // scene.viewProj, scene.lightDir, scene.ambient, scene.lightColor, scene.lightViewProj
    // objectData.model, objectData.color

    struct VSOut {
      @builtin(position) pos: vec4f,
      @location(0) normal: vec3f,
    }

    @vertex fn vs(@location(0) position: vec3f, @location(1) normal: vec3f) -> VSOut {
      var out: VSOut;
      out.pos = scene.viewProj * objectData.model * vec4f(position, 1.0);
      out.normal = normalize((objectData.model * vec4f(normal, 0.0)).xyz);
      return out;
    }

    @fragment fn fs(in: VSOut) -> @location(0) vec4f {
      let ndotl = max(dot(in.normal, scene.lightDir.xyz), 0.0);
      return vec4f(objectData.color.rgb * ndotl, 1.0);
    }
  `,
  color: 0xff0000,
  uniforms: new Float32Array(4), // optional, available at @group(2)
})
```

The preamble provides these bindings:

| Group | Binding | Type                 | Content                                                              |
| ----- | ------- | -------------------- | -------------------------------------------------------------------- |
| 0     | 0       | `uniform Scene`      | viewProj, lightDir, ambient, lightColor, lightViewProj, shadowParams |
| 0     | 1       | `texture_depth_2d`   | Shadow map                                                           |
| 0     | 2       | `sampler_comparison` | Shadow sampler                                                       |
| 1     | 0       | `storage ObjectData` | Per-object model matrix and color                                    |
| 2     | 0       | (yours)              | Custom uniforms Float32Array                                         |

### Lights

```typescript
const ambient = new AmbientLight(0x404040, 0.5)
scene.add(ambient)

const dir = new DirectionalLight(0xffffff, 1.2)
dir.position.set(5, 10, 7)
dir.castShadow = true
dir.shadow.mapSize.set(2048, 2048)
dir.shadow.camera.near = 0.5
dir.shadow.camera.far = 200
dir.shadow.camera.left = -60 // orthographic shadow frustum bounds
scene.add(dir)
```

### Textures

```typescript
import { loadTexture } from 'nanothree'

const texture = loadTexture('/textures/grass.png', tex => {
  // texture is ready, material will auto-update on next render
})

const material = new MeshLambertMaterial({ color: 0xffffff, map: texture })
```

Textures are cached by URL. Call `clearTextureCache()` to dispose all.

### GLTF/GLB Loading

Draco and Basis decoders are **bundled with nanothree** and loaded on demand. No setup required:

```typescript
import { GLTFLoader, AnimationMixer } from 'nanothree'

const loader = new GLTFLoader()
loader.load('/models/character.glb', result => {
  scene.add(result.scene)

  // Play animation
  if (result.animations.length > 0) {
    const mixer = new AnimationMixer(result.scene)
    const action = mixer.clipAction(result.animations[0])
    action.play()
    // call mixer.update(dt) each frame
  }
})
```

To self-host the decoder files instead of using the bundled ones:

```typescript
loader.setDracoDecoderPath('/decoders/draco/')
loader.setBasisTranscoderPath('/decoders/basis/')
```

**Supported GLTF features**: meshes, materials (base color + texture), node hierarchy, animations (translation/rotation/scale), skeletal meshes (skins, joints, weights), Draco compression (`KHR_draco_mesh_compression`), KTX2 textures (`KHR_texture_basisu`, `EXT_texture_webp`, `EXT_texture_avif`).

**Not supported**: morph targets, cameras, lights, sparse accessors, PBR (metalness/roughness/normal maps).

### Animation

```typescript
const mixer = new AnimationMixer(model)
const action = mixer.clipAction(clip)

action.play()
action.stop()
action.reset()
action.setLoop(true)
action.clampWhenFinished = true
action.fadeIn(0.3) // crossfade in over 0.3s
action.fadeOut(0.3) // crossfade out

// Find clip by name
const clip = AnimationClip.findByName(result.animations, 'Walk')

// Each frame
mixer.update(deltaTime)
mixer.stopAllAction()
```

### Instancing

**`InstancedMesh`** - Render many copies of the same geometry with per-instance transforms and colors:

```typescript
const instances = new InstancedMesh(geometry, material, 1000)
const matrix = new Float32Array(16)
instances.setMatrixAt(i, matrix)
instances.setColorAt(i, new Color(0xff0000))
scene.add(instances)
```

**`InstancedSprite`** - Lightweight GPU-billboarded particles:

```typescript
const sprites = new InstancedSprite(500, NormalBlending)
sprites.setPositionAt(i, x, y, z)
sprites.setSizeAt(i, 0.5)
sprites.setColorAt(i, new Color(1, 1, 0))
sprites.setAlphaAt(i, 0.8)
scene.add(sprites)
```

### Raycasting

```typescript
const raycaster = new Raycaster()

// From screen coordinates (NDC: -1 to 1)
raycaster.setFromCamera([ndcX, ndcY], camera)

// Or from world-space ray
raycaster.set(origin, direction)

const hits = raycaster.intersectObject(scene, true) // recursive
// hits[0] = { distance, point: [x, y, z], object }
```

### Frustum Culling

Automatic when you pass the camera's view-projection to `updateMatrixWorld`:

```typescript
camera.updateViewProjection()
scene.updateMatrixWorld(camera.viewProjection)
renderer.render(scene, camera)
```

Objects outside the camera frustum are skipped during rendering. Bounding spheres are computed automatically from geometry.

### Shadow Mapping

```typescript
renderer.shadowMap.enabled = true

const light = new DirectionalLight(0xffffff, 1)
light.castShadow = true
light.shadow.mapSize.set(2048, 2048) // shadow map resolution

const mesh = new Mesh(geometry, material)
mesh.castShadow = true
mesh.receiveShadow = true
```

Shadows use a depth pass from the light's perspective with 4-tap PCF filtering.

## Three.js API Differences

nanothree follows Three.js naming conventions but differs in several ways:

| Area                | Three.js                                            | nanothree                                                     |
| ------------------- | --------------------------------------------------- | ------------------------------------------------------------- |
| **Backend**         | WebGL2 + WebGPU                                     | WebGPU only                                                   |
| **Custom shaders**  | TSL (Three Shading Language) or GLSL                | **WGSL** via `ShaderMaterial`                                 |
| **Materials**       | MeshStandardMaterial (PBR), many others             | `MeshLambertMaterial` (diffuse only), `MeshBasicMaterial`     |
| **Lights**          | Ambient, Directional, Point, Spot, Hemisphere, Area | `AmbientLight`, `DirectionalLight` only                       |
| **Renderer init**   | Synchronous constructor                             | `new WebGPURenderer({ canvas })` then `await renderer.init()` |
| **Textures**        | `TextureLoader` class                               | `loadTexture(url)` function                                   |
| **GLTF decoders**   | Manual setup required                               | **Bundled**, zero config (Draco + Basis)                      |
| **Post-processing** | EffectComposer, render passes                       | Not supported                                                 |
| **Morph targets**   | Supported                                           | Not supported                                                 |
| **Point lights**    | Supported                                           | Not supported                                                 |
| **PBR**             | metalness/roughness/normal/emissive maps            | Not supported (Lambert only)                                  |
| **Orbit controls**  | `OrbitControls` class                               | Not included (see example for manual implementation)          |
| **React bindings**  | `@react-three/fiber`                                | Not included                                                  |

## Bundle Size

The library builds into three chunks, loaded on demand:

| Chunk            | Size    | Loaded when                 |
| ---------------- | ------- | --------------------------- |
| Main library     | ~198 KB | Always                      |
| Draco decoder    | ~329 KB | GLTF with Draco compression |
| Basis transcoder | ~768 KB | GLTF with KTX2 textures     |

## License

MIT
