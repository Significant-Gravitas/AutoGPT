/**
 * Copies the Silero VAD model and its onnxruntime WASM into `public/vad/`.
 *
 * `@ricky0123/vad-web` otherwise fetches them from a public CDN at runtime,
 * which would put a third-party request on the copilot page.
 */
import { createRequire } from "node:module";
import { copyFile, mkdir } from "node:fs/promises";
import path from "node:path";

const DESTINATION = path.join(process.cwd(), "public", "vad");

const fromProject = createRequire(import.meta.url);
// onnxruntime is vad-web's dependency, not ours, so pnpm's strict layout only
// exposes it from inside that package — and its `exports` map hides `dist/`,
// hence resolving the entry point and walking up to the directory.
const fromVad = createRequire(fromProject.resolve("@ricky0123/vad-web"));
const ortDist = path.dirname(fromVad.resolve("onnxruntime-web"));

const ASSETS = [
  fromProject.resolve("@ricky0123/vad-web/dist/silero_vad_v5.onnx"),
  fromProject.resolve("@ricky0123/vad-web/dist/vad.worklet.bundle.min.js"),
  path.join(ortDist, "ort-wasm-simd-threaded.wasm"),
  path.join(ortDist, "ort-wasm-simd-threaded.mjs"),
];

await mkdir(DESTINATION, { recursive: true });
await Promise.all(
  ASSETS.map((source) =>
    copyFile(source, path.join(DESTINATION, path.basename(source))),
  ),
);
