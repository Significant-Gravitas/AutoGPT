import { readdirSync, readFileSync } from "node:fs";
import { createRequire } from "node:module";
import path from "node:path";
import { describe, expect, it } from "vitest";
import { reactCompilerSources } from "../next.config.mjs";

// Every file the React Compiler compiles (`reactCompilerSources` in
// next.config.mjs), run through the compiler with Next's own bundled Babel,
// the same pipeline as `next dev` and `next build`. A function the compiler
// cannot compile ships un-memoised without any error; this test is what
// notices. Lint misses some of these, such as `try/finally` or a `throw`
// inside `try`.

/**
 * Functions the compiler refuses today, one entry per function. The list can
 * only shrink: the test fails on a bailout that is not listed here, and on an
 * entry whose function now compiles, so a fixed bailout must be removed.
 */
const KNOWN_BAILOUTS: { file: string; category: string; why: string }[] = [
  {
    file: "src/app/(platform)/copilot/components/ChatMessagesContainer/components/VirtualTranscript/useVirtualTranscript.ts",
    category: "IncompatibleLibrary",
    why: "TanStack Virtual's useVirtualizer returns methods that would go stale if memoised.",
  },
  {
    file: "src/app/(platform)/copilot/components/ChatMessagesContainer/components/VirtualTranscript/useQuietRows.ts",
    category: "Refs",
    why: "Render-time bookkeeping of which rows in the virtual window animate.",
  },
];

const require = createRequire(import.meta.url);

interface CompilerEvent {
  kind: string;
  fnLoc?: { start?: { line: number; column: number } } | null;
  detail?: { category?: string; options?: { category?: string } };
}

const babel = require("next/dist/compiled/babel/core") as {
  transformSync: (code: string, options: object) => unknown;
};
const syntaxJsx = require("next/dist/compiled/babel/plugin-syntax-jsx");
const presetTypescript = require("next/dist/compiled/babel/preset-typescript");
const reactCompiler = require("babel-plugin-react-compiler");

const FRONTEND_ROOT = path.resolve(__dirname, "..");
const BAILOUT_KINDS = new Set(["CompileError", "CompileSkip", "PipelineError"]);

function toPosix(file: string) {
  return file.split(path.sep).join("/");
}

function isCompiledSource(file: string) {
  const posix = `/${file}`;
  return (
    reactCompilerSources.some((source) => posix.includes(source)) &&
    /\.(js|jsx|ts|tsx)$/.test(file) &&
    !file.endsWith(".d.ts") &&
    !/\.(test|spec|stories)\.(js|jsx|ts|tsx)$/.test(file) &&
    !file.split("/").includes("__tests__")
  );
}

function listCompiledFiles() {
  return readdirSync(path.join(FRONTEND_ROOT, "src"), { recursive: true })
    .map((file) => `src/${toPosix(String(file))}`)
    .filter(isCompiledSource)
    .sort();
}

function compilerEvents(file: string, source: string) {
  const events: CompilerEvent[] = [];
  babel.transformSync(source, {
    filename: path.join(FRONTEND_ROOT, file),
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
          logger: {
            logEvent: (_filename: string | null, event: CompilerEvent) =>
              events.push(event),
          },
        },
      ],
    ],
    presets: [[presetTypescript, { allowNamespaces: true }]],
  });
  return events;
}

function category(event: CompilerEvent) {
  return (
    event.detail?.category ?? event.detail?.options?.category ?? event.kind
  );
}

/** One `file category` key per function that bailed out. */
function bailoutsIn(file: string, source: string) {
  const byFunction = new Map<string, Set<string>>();
  for (const event of compilerEvents(file, source)) {
    if (!BAILOUT_KINDS.has(event.kind)) continue;
    const start = event.fnLoc?.start;
    const fn = start ? `${start.line}:${start.column}` : event.kind;
    const categories = byFunction.get(fn) ?? new Set<string>();
    categories.add(category(event));
    byFunction.set(fn, categories);
  }
  return [...byFunction.values()].map(
    (categories) => `${file} ${[...categories].sort().join("+")}`,
  );
}

function bailoutKey({ file, category }: { file: string; category: string }) {
  return `${file} ${category}`;
}

function difference(from: string[], remove: string[]) {
  const left = [...remove];
  return from.filter((key) => {
    const index = left.indexOf(key);
    if (index === -1) return true;
    left.splice(index, 1);
    return false;
  });
}

describe("React Compiler coverage", () => {
  const files = listCompiledFiles();
  const actual = files
    .flatMap((file) =>
      bailoutsIn(file, readFileSync(path.join(FRONTEND_ROOT, file), "utf8")),
    )
    .sort();
  const known = KNOWN_BAILOUTS.map(bailoutKey).sort();

  it("finds the compiled folders", () => {
    for (const source of reactCompilerSources) {
      expect(
        files.some((file) => `/${file}`.includes(source)),
        `no compiled files under ${source}`,
      ).toBe(true);
    }
  });

  it("reports no bailout that is not in KNOWN_BAILOUTS", () => {
    expect(
      difference(actual, known),
      "New React Compiler bailouts. Fix the function so the compiler can compile it; add it to KNOWN_BAILOUTS only if the compiler refuses it on purpose.",
    ).toEqual([]);
  });

  it("lists only bailouts that still happen, so the list can only shrink", () => {
    expect(
      difference(known, actual),
      "These functions compile now. Remove them from KNOWN_BAILOUTS.",
    ).toEqual([]);
  });

  it("catches a try/finally that lint does not flag", () => {
    const file = "src/app/(platform)/copilot/usePlantedBailout.ts";
    const source = `
      import { useState } from "react";
      export function usePlantedBailout() {
        const [value] = useState(0);
        try {
          return value + 1;
        } finally {
          console.log("done");
        }
      }
    `;

    expect(bailoutsIn(file, source)).toHaveLength(1);
  });
});
