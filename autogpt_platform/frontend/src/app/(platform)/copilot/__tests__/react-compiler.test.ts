import { readdirSync, readFileSync } from "node:fs";
import { createRequire } from "node:module";
import path from "node:path";
import { describe, expect, it } from "vitest";
import { reactCompilerSources } from "../../../../../next.config.mjs";

// The same Babel pipeline Next's React Compiler loader runs, from Next's own
// bundled Babel, so this sees what `next dev` and `next build` see.
const require = createRequire(import.meta.url);

interface CompilerEvent {
  kind: string;
  fnName?: string | null;
  reason?: string;
  detail?: { reason?: string; options?: { reason?: string } };
}

interface BabelResult {
  code: string | null;
}

const babel = require("next/dist/compiled/babel/core") as {
  transformSync: (code: string, options: object) => BabelResult | null;
};
const syntaxJsx = require("next/dist/compiled/babel/plugin-syntax-jsx");
const presetTypescript = require("next/dist/compiled/babel/preset-typescript");
const reactCompiler = require("babel-plugin-react-compiler");

const FRONTEND_ROOT = path.resolve(__dirname, "../../../../..");
const COMPILED_DIRS = [
  "src/app/(platform)/copilot",
  "src/components/organisms/Conversation",
  "src/components/organisms/Message",
  "src/components/organisms/PromptInput",
];

/**
 * Left to the compiler's own judgement. TanStack Virtual's `useVirtualizer`
 * returns methods that would go stale if memoised, so the compiler refuses
 * the hook that calls it; `useQuietRows` is that hook's render-time
 * bookkeeping of which rows in the virtual window animate.
 */
const EXPECTED_BAILOUTS: Record<string, string> = {
  "components/ChatMessagesContainer/components/VirtualTranscript/useVirtualTranscript.ts":
    "IncompatibleLibrary",
  "components/ChatMessagesContainer/components/VirtualTranscript/useQuietRows.ts":
    "Refs",
};

function listSourceFiles(dir: string): string[] {
  return readdirSync(path.join(FRONTEND_ROOT, dir), { recursive: true })
    .map(String)
    .filter(
      (file) =>
        /\.(ts|tsx)$/.test(file) &&
        !file.endsWith(".d.ts") &&
        !/\.(test|spec|stories)\.tsx?$/.test(file) &&
        !file.split(path.sep).includes("__tests__"),
    )
    .map((file) => path.join(FRONTEND_ROOT, dir, file));
}

function compile(file: string) {
  const events: CompilerEvent[] = [];
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
          logger: {
            logEvent: (_filename: string | null, event: CompilerEvent) =>
              events.push(event),
          },
        },
      ],
    ],
    presets: [[presetTypescript, { allowNamespaces: true }]],
  });
  return { events, code: result?.code ?? "" };
}

function category(event: CompilerEvent) {
  const detail = event.detail as
    | { category?: string; options?: { category?: string } }
    | undefined;
  return detail?.category ?? detail?.options?.category ?? event.kind;
}

describe("React Compiler on the copilot", () => {
  const files = COMPILED_DIRS.flatMap(listSourceFiles);
  const results = files.map((file) => ({ file, ...compile(file) }));

  it("compiles every copilot component and hook", () => {
    const failures = results.flatMap(({ file, events }) => {
      const relative = path.relative(
        path.join(FRONTEND_ROOT, "src/app/(platform)/copilot"),
        file,
      );
      return events
        .filter((event) =>
          ["CompileError", "CompileSkip", "PipelineError"].includes(event.kind),
        )
        .filter((event) => EXPECTED_BAILOUTS[relative] !== category(event))
        .map(
          (event) =>
            `${path.relative(FRONTEND_ROOT, file)}: ${event.kind} ${category(event)}`,
        );
    });

    expect(failures).toEqual([]);
  });

  it("still reports the expected bailouts, so stale entries get removed", () => {
    for (const [file, expected] of Object.entries(EXPECTED_BAILOUTS)) {
      const result = results.find(({ file: absolute }) =>
        absolute.endsWith(path.join("copilot", file)),
      );
      expect(result?.events.map(category)).toContain(expected);
    }
  });

  it("memoises the page and stream hooks", () => {
    const compiled = results.flatMap(({ events }) =>
      events
        .filter((event) => event.kind === "CompileSuccess")
        .map((event) => event.fnName),
    );

    expect(compiled).toEqual(
      expect.arrayContaining([
        "useCopilotPage",
        "useCopilotRuntimePage",
        "useCopilotStream",
        "useCopilotRuntimeStream",
        "CopilotChatHost",
      ]),
    );
  });

  // Vitest runs the `react` package, but Next serves its own vendored React to
  // the app, which can lag behind it (no `useEffectEvent` in Next 15.5's).
  it("only imports React APIs the React that Next ships has", () => {
    const nextReact = require("next/dist/compiled/react") as Record<
      string,
      unknown
    >;
    const missing = files.flatMap((file) =>
      [
        ...readFileSync(file, "utf8").matchAll(
          /import\s*\{([^}]*)\}\s*from\s*"react"/g,
        ),
      ]
        .flatMap((match) => match[1].split(","))
        .map((name) => name.trim().split(/\s+as\s+/)[0])
        .filter((name) => /^use[A-Z]|^[a-z]/.test(name))
        .filter((name) => !name.startsWith("type ") && !(name in nextReact))
        .map((name) => `${path.relative(FRONTEND_ROOT, file)}: ${name}`),
    );

    expect(missing).toEqual([]);
  });

  it("leaves files outside the copilot uncompiled", () => {
    const outside = path.join(
      FRONTEND_ROOT,
      "src/components/atoms/Button/Button.tsx",
    );
    const { events, code } = compile(outside);

    expect(events.filter((event) => event.kind !== "Timing")).toEqual([]);
    expect(code).not.toContain("react/compiler-runtime");
  });
});
