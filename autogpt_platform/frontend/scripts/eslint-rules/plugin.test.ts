// @vitest-environment node
import { readFileSync } from "node:fs";
import path from "node:path";
import { RuleTester } from "oxlint/plugins-dev";
import { describe, it } from "vitest";
import plugin, { MANUAL_MEMO_BASELINE_FILE } from "./plugin.mjs";

RuleTester.describe = describe;
RuleTester.it = it;

type TesterRule = Parameters<RuleTester["run"]>[1];

const FRONTEND_ROOT = path.resolve(__dirname, "../..");
const COPILOT_FILE = "src/app/(platform)/copilot/Fixture.tsx";

const tester = new RuleTester({
  cwd: FRONTEND_ROOT,
  languageOptions: { sourceType: "module" },
});

tester.run(
  "no-hookless-use-function",
  plugin.rules["no-hookless-use-function"] as unknown as TesterRule,
  {
    valid: [
      "function useCount() { const [count] = useState(0); return count; }",
      "export const useTheme = () => React.useContext(ThemeContext);",
      "function useSession(promise) { return use(promise); }",
      "function useRow() { return useOtherHook(); }",
      // Not hook-named, so the rule has nothing to say.
      "function formatCount(count) { return String(count); }",
      "function user() { return 1; }",
      "const useStore = create(() => ({}));",
      // Hook-named on purpose and kept out of the compiler.
      'function useReadEveryRender(read) { "use no memo"; return read(); }',
    ],
    invalid: [
      {
        code: "export function useLabel(count) { return `${count} items`; }",
        errors: [{ message: /`useLabel` calls no hook/ }],
      },
      {
        code: "const useLabel = (count) => String(count);",
        errors: [{ message: /`useLabel` calls no hook/ }],
      },
      {
        code: "const useLabel = function (count) { return String(count); };",
        errors: [{ message: /`useLabel` calls no hook/ }],
      },
      {
        // A hook called inside a nested callback belongs to that callback.
        code: "function useLater() { return () => useState(0); }",
        errors: [{ message: /`useLater` calls no hook/ }],
      },
    ],
  },
);

const [baselineFile, baselineCount] = Object.entries(
  JSON.parse(readFileSync(MANUAL_MEMO_BASELINE_FILE, "utf8")) as Record<
    string,
    number
  >,
)[0];

function useMemoCalls(count: number) {
  return Array.from(
    { length: count },
    (_, index) => `const value${index} = useMemo(() => ${index}, []);`,
  ).join("\n");
}

const importMemo = `import { memo, useCallback, useMemo } from "react";\n`;

tester.run(
  "no-manual-memo",
  plugin.rules["no-manual-memo"] as unknown as TesterRule,
  {
    valid: [
      { code: "function Row() { return null; }", filename: COPILOT_FILE },
      {
        code: `${importMemo}// memo-boundary: one row per message; the transcript re-renders on every token.\nexport const Row = memo(function Row() { return null; });`,
        filename: COPILOT_FILE,
      },
      {
        // Only React's memo APIs count, not a local function with the name.
        code: "function useMemo(fn) { return fn(); }\nuseMemo(() => 1);",
        filename: COPILOT_FILE,
      },
      {
        // A file written before the compiler keeps its existing count.
        code: `${importMemo}${useMemoCalls(baselineCount)}`,
        filename: baselineFile,
      },
    ],
    invalid: [
      {
        code: `${importMemo}const value = useMemo(() => 1, []);`,
        filename: COPILOT_FILE,
        errors: [{ message: /`useMemo` is not needed here/ }],
      },
      {
        code: `${importMemo}const onClick = useCallback(() => {}, []);`,
        filename: COPILOT_FILE,
        errors: [{ message: /`useCallback` is not needed here/ }],
      },
      {
        code: `import * as React from "react";\nconst value = React.useMemo(() => 1, []);`,
        filename: COPILOT_FILE,
        errors: [{ message: /`useMemo` is not needed here/ }],
      },
      {
        code: `import { useMemo as memoise } from "react";\nconst value = memoise(() => 1, []);`,
        filename: COPILOT_FILE,
        errors: [{ message: /`useMemo` is not needed here/ }],
      },
      {
        code: `${importMemo}export const Row = memo(function Row() { return null; });`,
        filename: COPILOT_FILE,
        errors: [{ message: /memo-boundary/ }],
      },
      {
        // The comment belongs to the outer function, not to the memo() call.
        code: `${importMemo}// memo-boundary: one row per message.\nexport function makeRow() {\n  return memo(function Row() { return null; });\n}`,
        filename: COPILOT_FILE,
        errors: [{ message: /memo-boundary/ }],
      },
      {
        code: `function Row() {\n  "use no memo";\n  return null;\n}`,
        filename: COPILOT_FILE,
        errors: [{ message: /"use no memo" overrides/ }],
      },
      {
        code: `"use memo";\nexport function Row() { return null; }`,
        filename: COPILOT_FILE,
        errors: [{ message: /"use memo" overrides/ }],
      },
      {
        code: `${importMemo}${useMemoCalls(baselineCount + 1)}`,
        filename: baselineFile,
        errors: Array.from({ length: baselineCount + 1 }, () => ({
          message: new RegExp(
            `allowed ${baselineCount} existing manual memo call\\(s\\) and now has ${baselineCount + 1}`,
          ),
        })),
      },
      {
        code: `${importMemo}${useMemoCalls(baselineCount - 1)}`,
        filename: baselineFile,
        errors: [{ message: /Lower its entry/ }],
      },
    ],
  },
);
