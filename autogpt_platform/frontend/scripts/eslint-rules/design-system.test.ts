// @vitest-environment node
import { spawnSync } from "node:child_process";
import {
  mkdirSync,
  mkdtempSync,
  readFileSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { RuleTester } from "oxlint/plugins-dev";
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import plugin, {
  colorAdvice,
  DESIGN_SYSTEM_BASELINE_FILE,
} from "./design-system.mjs";
import { countDiagnostics } from "./update-design-system-baseline.mjs";

RuleTester.describe = describe;
RuleTester.it = it;

type TesterRule = Parameters<RuleTester["run"]>[1];

const FRONTEND_ROOT = path.resolve(__dirname, "../..");
// Not in design-system-baseline.json, so it is allowed nothing.
const FILE = "src/app/(platform)/fixture/Fixture.tsx";

const tester = new RuleTester({
  cwd: FRONTEND_ROOT,
  languageOptions: {
    sourceType: "module",
    parserOptions: { ecmaFeatures: { jsx: true } },
  },
});

function rule(name: keyof typeof plugin.rules) {
  return plugin.rules[name] as unknown as TesterRule;
}

function jsx(className: string) {
  return { code: `<div className="${className}" />`, filename: FILE };
}

describe("colorAdvice", () => {
  it("names the token with the same colour, keeping variants and opacity", () => {
    expect(colorAdvice("text-zinc-500")).toMatch(
      /Use "text-foreground-tertiary" instead/,
    );
    expect(colorAdvice("hover:bg-zinc-100/50")).toMatch(
      /Use "hover:bg-surface-hover\/50" instead/,
    );
    expect(colorAdvice("text-[#83838C]")).toMatch(
      /Use "text-foreground-tertiary" instead/,
    );
    expect(colorAdvice("bg-[rgb(239,239,240)]")).toMatch(
      /Use "bg-surface-hover" instead/,
    );
  });

  it("lists the roles for the utility when no token has the colour", () => {
    const advice = colorAdvice("text-rose-700");
    expect(advice).toMatch(/No text- token has this exact colour/);
    expect(advice).toMatch(/foreground-secondary/);
    expect(advice).not.toMatch(/surface-hover/);
  });

  it("tells a dark: colour to go", () => {
    expect(colorAdvice("dark:bg-zinc-800")).toMatch(/never renders: delete it/);
  });

  it("ignores what is not a colour", () => {
    expect(colorAdvice("text-[13px]")).toBeNull();
    expect(colorAdvice("font-[600]")).toBeNull();
  });
});

tester.run("no-raw-colors", rule("no-raw-colors"), {
  valid: [jsx("bg-surface text-foreground-secondary border-border-base")],
  invalid: [
    {
      ...jsx("text-zinc-500"),
      errors: [
        {
          message:
            /^"text-zinc-500" uses the raw Tailwind palette\. Use "text-foreground-tertiary" instead.*See docs\/engineering\/design-system\.md\.$/,
        },
      ],
    },
  ],
});

tester.run("no-arbitrary-values", rule("no-arbitrary-values"), {
  valid: [jsx("bg-surface")],
  invalid: [
    {
      ...jsx("bg-[#EFEFF0]"),
      errors: [{ message: /hardcodes a colour\. Use "bg-surface-hover"/ }],
    },
    {
      ...jsx("font-[600]"),
      errors: [{ message: /hardcodes a value\. Use "font-semibold"/ }],
    },
  ],
});

tester.run("no-dark-variant", rule("no-dark-variant"), {
  valid: [
    jsx("bg-surface data-[theme=dark]:hidden"),
    { code: `const theme = { dark: "x" };`, filename: FILE },
  ],
  invalid: [
    {
      ...jsx("bg-surface dark:bg-zinc-800 hover:dark:text-white"),
      errors: [
        { message: /^"dark:bg-zinc-800" is a dark: class/ },
        { message: /^"hover:dark:text-white" is a dark: class/ },
      ],
    },
    {
      code: "const styles = cn(`light:text-black ${extra}`);",
      filename: FILE,
      errors: [{ message: /"light:text-black" is a light: class/ }],
    },
  ],
});

tester.run("no-legacy-import", rule("no-legacy-import"), {
  valid: [
    {
      code: `import { Button } from "@/components/atoms/Button/Button";\nimport { readLegacyCookie } from "./legacy-cookies";`,
      filename: FILE,
    },
  ],
  invalid: [
    {
      code: `import { Sidebar } from "@/components/__legacy__/Sidebar";`,
      filename: FILE,
      errors: [{ message: /is a legacy component/ }],
    },
    {
      code: `const Lazy = import("../legacy/Panel");\nexport * from "@/components/ui/button";`,
      filename: FILE,
      errors: [
        { message: /is a legacy component/ },
        { message: /raw shadcn primitives, which are gone/ },
      ],
    },
    {
      code: `import { LegacyButton } from "@/components/atoms/Button/LegacyButton";`,
      filename: FILE,
      errors: [
        { message: /Render <Button> from @\/components\/atoms\/Button/ },
      ],
    },
    {
      code: `import { legacyButtonVariants } from "@/components/atoms/Button/helpers";`,
      filename: FILE,
      errors: [
        { message: /legacyButtonVariants is internal to the Button atom/ },
      ],
    },
  ],
});

tester.run("no-legacy-button-variant", rule("no-legacy-button-variant"), {
  valid: [`<Button variant="secondary" />`, `const id = "legacy-session";`],
  invalid: [
    {
      code: `<Button variant="legacy-outline" size="legacy-sm" />`,
      filename: FILE,
      errors: [
        { message: /Button with variant="outline"/ },
        { message: /Button with size="small"/ },
      ],
    },
  ],
});

tester.run("hugeicons-only", rule("hugeicons-only"), {
  valid: [
    {
      code: `import { Delete02Icon } from "@hugeicons/core-free-icons";\nimport type { IconSvgElement } from "@hugeicons/react";`,
      filename: FILE,
    },
  ],
  invalid: [
    {
      code: `import { X } from "lucide-react";\nimport { Gear } from "@phosphor-icons/react/dist/ssr";`,
      filename: FILE,
      errors: [
        { message: /"lucide-react" is not the design-system icon set/ },
        { message: /"@phosphor-icons\/react\/dist\/ssr" is not/ },
      ],
    },
    {
      code: `import { HugeiconsIcon } from "@hugeicons/react";`,
      filename: FILE,
      errors: [{ message: /Do not render HugeiconsIcon directly/ }],
    },
  ],
});

tester.run("require-suppression-reason", rule("require-suppression-reason"), {
  valid: [
    `// oxlint-disable-next-line design-system/no-arbitrary-values -- xyflow positions handles in px\nconst a = 1;`,
    `/* eslint-disable-next-line no-console -- CLI output */\nconsole.log(1);`,
  ],
  invalid: [
    {
      code: `// eslint-disable-next-line no-console\nconsole.log(1);`,
      filename: FILE,
      errors: [{ message: /Say why this rule is suppressed/ }],
    },
  ],
});

// A file that design-system-baseline.json already lists keeps its count.
const [grandfatheredFile, grandfatheredCount] = Object.entries(
  JSON.parse(readFileSync(DESIGN_SYSTEM_BASELINE_FILE, "utf8")) as Record<
    string,
    Record<string, number>
  >,
)
  .map(([file, counts]) => [file, counts["no-dark-variant"] ?? 0] as const)
  .find(([, count]) => count > 0)!;

function darkClasses(count: number) {
  const classes = Array.from(
    { length: count },
    (_, index) => `dark:mt-${index}`,
  );
  return `<div className="${classes.join(" ")}" />`;
}

tester.run("no-dark-variant (baseline)", rule("no-dark-variant"), {
  valid: [
    { code: darkClasses(grandfatheredCount), filename: grandfatheredFile },
  ],
  invalid: [
    {
      code: darkClasses(grandfatheredCount + 1),
      filename: grandfatheredFile,
      errors: Array.from({ length: grandfatheredCount + 1 }, () => ({
        message: new RegExp(
          `allowed ${grandfatheredCount} existing design-system/no-dark-variant violation\\(s\\).*now has ${grandfatheredCount + 1}`,
        ),
      })),
    },
  ],
});

describe("countDiagnostics", () => {
  it("counts design-system errors per file and rule, sorted", () => {
    expect(
      countDiagnostics([
        { code: "design-system(no-raw-colors)", filename: "src/b.tsx" },
        { code: "design-system(no-raw-colors)", filename: "src/b.tsx" },
        { code: "design-system(no-dark-variant)", filename: "src/b.tsx" },
        { code: "design-system(no-dark-variant)", filename: "src/a.tsx" },
        { code: "design-system(baseline-outdated)", filename: "src/a.tsx" },
        { code: "autogpt(no-manual-memo)", filename: "src/a.tsx" },
      ]),
    ).toEqual({
      "src/a.tsx": { "no-dark-variant": 1 },
      "src/b.tsx": { "no-dark-variant": 1, "no-raw-colors": 2 },
    });
  });
});

// The whole mechanism through the oxlint CLI, as `pnpm lint` runs it: a
// fixture project with its own config and baseline.
describe("the ratchet, end to end", () => {
  let root: string;

  function write(file: string, content: string) {
    const full = path.join(root, file);
    mkdirSync(path.dirname(full), { recursive: true });
    writeFileSync(full, content);
  }

  function key(file: string) {
    return path
      .relative(FRONTEND_ROOT, path.join(root, file))
      .split(path.sep)
      .join("/");
  }

  function lint(baseline: Record<string, Record<string, number>>) {
    write("baseline.json", JSON.stringify(baseline));
    const result = spawnSync(
      path.join(FRONTEND_ROOT, "node_modules/.bin/oxlint"),
      ["-c", path.join(root, ".oxlintrc.json"), "-f", "unix", "src"],
      {
        cwd: root,
        encoding: "utf8",
        env: {
          ...process.env,
          DESIGN_SYSTEM_BASELINE_FILE: path.join(root, "baseline.json"),
          NODE_OPTIONS: "--disable-warning=MODULE_TYPELESS_PACKAGE_JSON",
        },
      },
    );
    return { status: result.status, output: result.stdout };
  }

  beforeAll(() => {
    root = mkdtempSync(path.join(tmpdir(), "design-system-lint-"));
    write(
      ".oxlintrc.json",
      JSON.stringify({
        categories: { correctness: "off" },
        jsPlugins: [
          path.join(FRONTEND_ROOT, "scripts/eslint-rules/design-system.mts"),
        ],
        rules: {
          "design-system/no-raw-colors": "error",
          "design-system/no-dark-variant": "error",
          "design-system/baseline-outdated": "warn",
        },
      }),
    );
    write(
      "src/Old.tsx",
      `export const Old = () => <div className="text-zinc-500 dark:bg-black" />;\n`,
    );
    write(
      "src/Clean.tsx",
      `export const Clean = () => <div className="text-foreground-tertiary" />;\n`,
    );
  });

  afterAll(() => rmSync(root, { recursive: true, force: true }));

  it("passes when every file is at its baseline", () => {
    const { status, output } = lint({
      [key("src/Old.tsx")]: { "no-raw-colors": 1, "no-dark-variant": 1 },
    });
    expect(output).not.toMatch(/design-system/);
    expect(status).toBe(0);
  });

  it("fails on a violation in a file the baseline does not list", () => {
    const { status, output } = lint({});
    expect(output).toMatch(
      /Old\.tsx.*"text-zinc-500" uses the raw Tailwind palette\. Use "text-foreground-tertiary" instead/,
    );
    expect(output).toMatch(/Old\.tsx.*"dark:bg-black" is a dark: class/);
    expect(status).toBe(1);
  });

  it("fails when one rule's count goes over, and names only that rule", () => {
    const { status, output } = lint({
      [key("src/Old.tsx")]: { "no-raw-colors": 1 },
    });
    expect(output).toMatch(/design-system\(no-dark-variant\)/);
    expect(output).not.toMatch(/design-system\(no-raw-colors\)/);
    expect(status).toBe(1);
  });

  it("passes with a warning when a file has fewer than its baseline", () => {
    const { status, output } = lint({
      [key("src/Old.tsx")]: { "no-raw-colors": 3, "no-dark-variant": 1 },
    });
    expect(output).toMatch(
      /Old\.tsx.*no-raw-colors 3 -> 1.*pnpm lint:ds-baseline.*design-system\(baseline-outdated\)/,
    );
    expect(status).toBe(0);
  });
});
