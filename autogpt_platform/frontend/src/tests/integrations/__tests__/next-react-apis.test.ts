// @vitest-environment node
import { readdirSync, readFileSync } from "node:fs";
import { createRequire } from "node:module";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { expect, test } from "vitest";

// Vitest and tsc use the installed `react` package, but the App Router renders
// with the React that Next vendors, which can lag behind it: Next 15.5's has no
// `useEffectEvent`. An API missing there passes every test and type check,
// then throws as the page renders. The copilot's compiler test checks its own
// folders; this checks everything the app ships.
const require = createRequire(import.meta.url);
const nextReact = require("next/dist/compiled/react") as Record<
  string,
  unknown
>;
const SRC = fileURLToPath(new URL("../../..", import.meta.url));

function shippedSources() {
  return readdirSync(SRC, { recursive: true })
    .map(String)
    .filter(
      (file) =>
        /\.(ts|tsx)$/.test(file) &&
        !file.endsWith(".d.ts") &&
        !/\.(test|spec|stories)\.tsx?$/.test(file) &&
        !file
          .split(path.sep)
          .some((part) => part === "__tests__" || part === "__mocks__"),
    )
    .map((file) => path.join(SRC, file));
}

test("the app imports only React APIs the React that Next ships has", () => {
  const missing = shippedSources().flatMap((file) =>
    [
      ...readFileSync(file, "utf8").matchAll(
        /import\s+(?:\w+\s*,\s*)?\{([^}]*)\}\s*from\s*["']react["']/g,
      ),
    ]
      .flatMap((match) => match[1].split(","))
      .map((name) => name.trim().split(/\s+as\s+/)[0])
      .filter((name) => /^use[A-Z]|^[a-z]/.test(name))
      .filter((name) => !name.startsWith("type ") && !(name in nextReact))
      .map((name) => `${path.relative(SRC, file)}: ${name}`),
  );

  expect(missing).toEqual([]);
});
