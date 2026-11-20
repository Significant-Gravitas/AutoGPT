import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import path from "node:path";
import { describe, expect, it } from "vitest";
import { reactCompilerSources } from "../../../../../../../next.config.mjs";

// Vitest runs the hook uncompiled, but `next dev`/`next build` compile this
// folder with the React Compiler, which caches a render-time read that has no
// reactive inputs for the life of the mount. A dismissal read from storage
// that way never hides the banner, so the hook must not read storage itself.
const require = createRequire(import.meta.url);

const babel = require("next/dist/compiled/babel/core") as {
  transformSync: (code: string, options: object) => { code: string } | null;
};
const syntaxJsx = require("next/dist/compiled/babel/plugin-syntax-jsx");
const presetTypescript = require("next/dist/compiled/babel/preset-typescript");
const reactCompiler = require("babel-plugin-react-compiler");

const HOOK = path.resolve(__dirname, "../useNotificationBanner.ts");

function compile(file: string) {
  const result = babel.transformSync(readFileSync(file, "utf8"), {
    filename: file,
    babelrc: false,
    configFile: false,
    plugins: [
      syntaxJsx,
      [
        reactCompiler,
        {
          compilationMode: "infer",
          sources: reactCompilerSources,
          panicThreshold: "none",
        },
      ],
    ],
    presets: [[presetTypescript, { allowNamespaces: true }]],
  });
  return result?.code ?? "";
}

describe("useNotificationBanner under the React Compiler", () => {
  it("is compiled, and reads its state from the store, not from storage", () => {
    const code = compile(HOOK);

    expect(code).toContain("react/compiler-runtime");
    expect(code).not.toContain("storage.get(");
    expect(code).not.toContain("@/services/storage");
  });
});
