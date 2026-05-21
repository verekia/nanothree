# CLAUDE.md

Guidance for future Claude sessions on this repo. Things you can't easily derive by reading the code.

## Repo layout

Monorepo with two workspaces:

- `library/` — the published `nanothree` package
- `example/` — Next.js demo app that imports the workspace library

## README is single-sourced

`README.md` at the repo root is canonical. `library/README.md` is overwritten on every build (`cp ../README.md README.md` in the library's `build` script). **Never edit `library/README.md` directly** — your changes will be wiped on the next publish.

## Build / validation commands

- `bun run typecheck` — root: `tsc -b --noEmit` across both workspaces. Library has its own `tsgo` typecheck via `library/package.json`.
- `bun run all` — the full pre-commit gauntlet: `format:check`, `lint` (oxlint), `typecheck`, `warden`, and `bun test`. Run this before declaring work done.
- `bun run --filter 'nanothree' build` (or `bun run pub`) — builds the library via tsup. Includes the decoder generation `prebuild` step.

## Bundled decoders are generated

`library/src/draco-inline.ts` and `library/src/basis-inline.ts` are **auto-generated** from binaries in `library/decoders/`. Don't edit them by hand. Regenerate with `bun run generate-decoders` (also runs automatically as a `prebuild` step). The base64-encoded WASM is what keeps GLTF loading zero-config for consumers.

## Pixel 10 / PowerVR texture workaround

Pixel 10 reports `adapter.info.vendor === 'img-tec'` and `architecture === 'd-series'` (Imagination Technologies PowerVR DXT). On that GPU, `device.queue.copyExternalImageToTexture` silently writes an all-zero texture — no error, no warning. Symptoms: textured meshes (especially skinned GLTF characters) vanish entirely, while shadows still animate (shadow pass doesn't sample textures).

`WebGPURenderer.init()` sets `_needsWriteTextureWorkaround` based on adapter info and plumbs it into `NanoTexture._ensureGPU(device, useWriteTexture)` (see `library/src/material.ts`). When the flag is set, we draw the source image to an `OffscreenCanvas`, read pixels via `getImageData`, and upload through `device.queue.writeTexture`.

If you add a new texture upload path, route it through `NanoTexture._ensureGPU` or replicate the same branch — direct `copyExternalImageToTexture` calls will silently break on Pixel 10. The standalone library [gputex](https://www.npmjs.com/package/gputex) implements the same workaround if you ever want to factor it out.

## Renderer architecture quirks

- Bind group conventions are documented in the "Custom Shaders" table in the README. The renderer reuses the `instanceLayout` (a storage buffer at `@group(2) @binding(0)`) for instanced meshes **and** for bone matrices on skinned meshes — same layout, different content.
- Per-frame bucket arrays on `WebGPURenderer` (`_solidMeshes`, `_skinnedTextured`, etc.) are reused and reset to length 0 each frame; do not assume they're empty between calls or allocate fresh ones.
- The textured fragment shaders emit `vec4f(color, texColor.a)` into an `src-alpha / one-minus-src-alpha` blended pipeline. This means any path that produces `alpha = 0` makes geometry invisible — relevant when debugging "nothing renders" bugs.
