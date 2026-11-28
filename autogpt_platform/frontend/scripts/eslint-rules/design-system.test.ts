// @vitest-environment node
import { spawnSync } from "node:child_process";
import {
  mkdirSync,
  mkdtempSync,
  readFileSync,
  realpathSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { RuleTester } from "oxlint/plugins-dev";
import { afterAll, beforeAll, describe, expect, it, vi } from "vitest";
import plugin, { colorAdvice } from "./design-system.mjs";
import {
  countDiagnostics,
  parseFailures,
} from "./update-design-system-baseline.mjs";

// The rules read their baseline when first used, so point them at a fixture
// before the plugin is imported: the real baseline shrinks to nothing as the
// migration lands.
const FIXTURE_BASELINE = vi.hoisted(() => {
  const file = `${__dirname}/__fixtures__/design-system-baseline.json`;
  process.env.DESIGN_SYSTEM_BASELINE_FILE = file;
  return file;
});

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
  valid: [
    jsx("bg-surface text-foreground-secondary border-border-base"),
    // The palette and hex colours are no-raw-palette-classes' and no-hex-colours'.
    jsx("text-zinc-500"),
    { code: `<path fill="#669CF6" />`, filename: FILE },
  ],
  invalid: [
    {
      code: `<path stroke="white" />`,
      filename: FILE,
      errors: [{ message: /^stroke="white" hardcodes a color/ }],
    },
  ],
});

tester.run("no-shadcn-alias-colours", rule("no-shadcn-alias-colours"), {
  valid: [
    jsx("bg-canvas text-foreground-tertiary border-border-base ring-focus"),
    // Roles that start with a shadcn name are not one.
    jsx("bg-accent-surface text-foreground-base border-border-strong"),
    { code: `const note = "accent-foreground";`, filename: FILE },
  ],
  invalid: [
    {
      ...jsx("bg-muted hover:text-muted-foreground/70 [&_svg]:border-border"),
      errors: [
        {
          message:
            /^"bg-muted" is a shadcn colour name.*Use "bg-surface-hover"/,
        },
        {
          message:
            /^"hover:text-muted-foreground\/70" .*Use "hover:text-foreground-tertiary\/70"/,
        },
        {
          message:
            /^"\[&_svg\]:border-border" .*Use "\[&_svg\]:border-border-base"/,
        },
      ],
    },
    {
      ...jsx("scrollbar-thin scrollbar-thumb-muted-foreground"),
      errors: [
        {
          message:
            /^"scrollbar-thumb-muted-foreground" .*Use "scrollbar-thumb-foreground-tertiary"/,
        },
      ],
    },
  ],
});

tester.run("no-raw-palette-classes", rule("no-raw-palette-classes"), {
  valid: [
    jsx("bg-orange/10 text-orange-foreground ring-sky/30"),
    // ds-colour-exempt.json: a user-picked palette.
    {
      code: `const swatch = "bg-rose-300";`,
      filename: "src/app/(platform)/raise/components/ColorStep/helpers.ts",
    },
    // ...or one class a file may keep.
    {
      code: `const dont = "text-blue-500";`,
      filename: "src/components/overview.stories.tsx",
    },
  ],
  invalid: [
    {
      ...jsx("text-orange-600 md:bg-black/80 shadow-white"),
      errors: [
        {
          message:
            /^"text-orange-600" uses the raw Tailwind palette\. Use "text-orange-foreground" \(scripts\/ds\/palette-map\.json\)/,
        },
        { message: /^"md:bg-black\/80" .*Use "md:bg-scrim\/80"/ },
        { message: /^"shadow-white" .*Use "shadow-surface"/ },
      ],
    },
    {
      code: `const other = "text-emerald-300";`,
      filename: "src/components/atoms/Text/Text.tsx",
      errors: [{ message: /"text-emerald-300" uses the raw Tailwind palette/ }],
    },
  ],
});

tester.run("no-hex-colours", rule("no-hex-colours"), {
  valid: [
    {
      code: `const colour = "rgb(var(--color-edge))"; const issue = "see PR #16483";`,
      filename: FILE,
    },
    {
      code: `const series = "#2a78d6";`,
      filename:
        "src/app/(platform)/team/[expertId]/components/ExpertOverview/helpers.ts",
    },
  ],
  invalid: [
    {
      code: 'const a = { color: "#fff" }; const b = `border:1px solid #E4E4E7`;',
      filename: FILE,
      errors: [
        { message: /^"#fff" hardcodes a colour/ },
        { message: /^"#E4E4E7" hardcodes a colour/ },
      ],
    },
    {
      // `bg-[#123456]` and `[color:#fff]` are no-arbitrary-colour's; the
      // same hex in the mask is still this rule's.
      code: `<div className="bg-[#123456] [color:#fff] [mask:linear-gradient(#123456_0,#fff,#0000)]" />`,
      filename: FILE,
      errors: [
        { message: /"#123456"/ },
        { message: /"#fff"/ },
        { message: /"#0000"/ },
      ],
    },
  ],
});

tester.run("no-colour-functions", rule("no-colour-functions"), {
  valid: [
    // A role read, with or without an alpha, and words that end in "rgb".
    {
      code: 'const a = "rgb(var(--color-edge))"; const b = `box-shadow:0 1px 2px rgb(var(--shadow-color) / 0.1)`; const c = "hsl(var(--x))"; const d = "brgb(1 2 3)";',
      filename: FILE,
    },
    jsx(
      "bg-[radial-gradient(120%_100%_at_50%_0%,rgb(var(--color-brand)/0.10),transparent_70%)]",
    ),
    // A class whose whole arbitrary value is the colour is
    // no-arbitrary-colour's.
    jsx("hover:text-[rgb(131,131,140)] ring-[oklch(0.7_0.1_200)]"),
    // ds-colour-exempt.json: a whole file, or one value a file may keep.
    {
      code: "const glass = `rgba(255,255,255,${tint})`;",
      filename: "src/components/molecules/GlassOrb/GlassSurface.tsx",
    },
    {
      code: `const grid = "<rect stroke='rgba(0,0,0,0.08)' />";`,
      filename: "src/app/(public)/tour/components/TourHero/TourBackground.tsx",
    },
  ],
  invalid: [
    {
      code: 'const a = { backgroundColor: "rgb(248 250 252)" }; const b = `border:1px solid hsla(0, 0%, 50%, 0.4)`;',
      filename: FILE,
      errors: [
        { message: /^"rgb\(248 250 252\)" hardcodes a colour\. Read a role/ },
        { message: /^"hsla\(0, 0%, 50%, 0\.4\)" hardcodes a colour/ },
      ],
    },
    {
      // Inside a gradient or a shadow, and a template literal's partial
      // colour.
      code: 'const wash = "bg-[linear-gradient(180deg,rgba(14,165,233,0.10),transparent)] shadow-[0_1px_2px_rgba(0,0,0,0.05)]"; const rim = `1px solid rgba(255,255,255,${edge})`;',
      filename: FILE,
      errors: [
        { message: /^"rgba\(14,165,233,0\.10\)"/ },
        { message: /^"rgba\(0,0,0,0\.05\)"/ },
        { message: /^"rgba\(255,255,255,"/ },
      ],
    },
    {
      // A signed first argument: CSS wraps a negative hue into range.
      code: 'const a = "hsl(-30 100% 50%)"; const b = "oklch( +0.7 0.1 200)"; const c = "rgb(-.5 0 0)";',
      filename: FILE,
      errors: [
        { message: /^"hsl\(-30 100% 50%\)" hardcodes a colour/ },
        { message: /^"oklch\( \+0\.7 0\.1 200\)" hardcodes a colour/ },
        { message: /^"rgb\(-\.5 0 0\)" hardcodes a colour/ },
      ],
    },
  ],
});

tester.run("no-arbitrary-colour", rule("no-arbitrary-colour"), {
  valid: [
    jsx("bg-surface text-[13px] border-[1.5px] bg-[url(/a.png)]"),
    // A role read through var(), and colours inside a shadow or gradient.
    jsx(
      "bg-[rgb(var(--color-surface))] shadow-[0_1px_2px_rgba(0,0,0,0.05)] bg-[linear-gradient(90deg,rgb(var(--color-surface-hover))_25%,transparent)]",
    ),
    // ds-colour-exempt.json: a user-picked palette, or one value a file may
    // keep, in any case.
    {
      code: `const note = "bg-[#ffc88a]";`,
      filename:
        "src/app/(platform)/build/components/FlowEditor/nodes/CustomNode/components/StickyNoteBlock.tsx",
    },
    {
      code: `const fallback = "bg-[#E2E8F0]";`,
      filename:
        "src/components/contextual/OutputRenderers/renderers/CodeRenderer.tsx",
    },
  ],
  invalid: [
    {
      ...jsx(
        "bg-[#EFEFF0] hover:text-[rgb(131,131,140)] border-[color:hsl(0_0%_50%)]/40 [color:#fff] ring-[oklch(0.7_0.1_200)]",
      ),
      errors: [
        {
          message:
            /^"bg-\[#EFEFF0\]" hardcodes a colour\. Use "bg-surface-hover" instead/,
        },
        {
          message:
            /^"hover:text-\[rgb\(131,131,140\)\]" .*Use "hover:text-foreground-tertiary" or "hover:text-placeholder" instead/,
        },
        { message: /^"border-\[color:hsl\(0_0%_50%\)\]\/40" hardcodes/ },
        { message: /^"ring-\[oklch\(0\.7_0\.1_200\)\]" hardcodes/ },
        { message: /^"\[color:#fff\]" hardcodes a colour/ },
      ],
    },
  ],
});

tester.run("no-misused-role", rule("no-misused-role"), {
  valid: [
    jsx(
      "bg-fill-subtle bg-scrim/20 text-foreground-inverse bg-surface-inverse text-foreground-on-color border-foreground-subtle ring-foreground-subtle/40",
    ),
    // A background clipped to its text, gradient or solid, is that text's
    // colour.
    jsx(
      "bg-gradient-to-r from-foreground-subtle via-foreground-secondary to-foreground-subtle bg-clip-text text-transparent",
    ),
    jsx("bg-foreground-body bg-clip-text text-transparent"),
    // The clip may sit in another static part of the same class list.
    {
      code: 'const a = <div className={`bg-clip-text text-transparent ${"from-foreground-body"} bg-gradient-to-r to-foreground-secondary`} />;',
      filename: FILE,
    },
    {
      code: 'const a = cn("bg-clip-text text-transparent", on && "from-foreground-body");',
      filename: FILE,
    },
    {
      code: 'const a = cva("bg-clip-text text-transparent", { variants: { tone: { soft: "from-foreground-subtle" } } });',
      filename: FILE,
    },
    // A clip under a variant covers a fill under the same variant or more.
    jsx("hover:bg-foreground-body hover:bg-clip-text text-transparent"),
    jsx("md:bg-clip-text md:hover:from-foreground-body text-transparent"),
    // Other words that start the same.
    jsx("text-surface-ish bg-foreground-ish"),
    {
      code: `// oxlint-disable-next-line design-system/no-misused-role -- waits for a design call\nconst a = "text-surface-muted";`,
      filename: FILE,
    },
  ],
  invalid: [
    {
      ...jsx(
        "bg-foreground-subtle hover:bg-foreground-base/20 from-foreground-tertiary to-foreground-emphasis",
      ),
      errors: [
        {
          message:
            /^"bg-foreground-subtle" paints a fill with a text role.*Use "bg-fill-subtle", the fill role with the same colour/,
        },
        {
          message:
            /^"hover:bg-foreground-base\/20" .*Use "bg-fill-strong".*A backdrop is "scrim"/,
        },
        { message: /^"from-foreground-tertiary" .*Use "from-fill-soft"/ },
        { message: /^"to-foreground-emphasis" .*Use a fill role/ },
      ],
    },
    {
      // The same background, not clipped to text, is a fill.
      ...jsx("bg-foreground-body text-transparent"),
      errors: [{ message: /^"bg-foreground-body" .*Use "bg-fill-bold"/ }],
    },
    {
      // Clipped only on hover: by default the text role is a fill.
      ...jsx("bg-foreground-body hover:bg-clip-text text-transparent"),
      errors: [{ message: /^"bg-foreground-body" paints a fill/ }],
    },
    {
      ...jsx("md:bg-clip-text from-foreground-body text-transparent"),
      errors: [{ message: /^"from-foreground-body" paints a fill/ }],
    },
    {
      code: 'const a = <div className={`text-transparent ${"from-foreground-body"} bg-gradient-to-r`} />;',
      filename: FILE,
      errors: [{ message: /^"from-foreground-body" paints a fill/ }],
    },
    {
      // A clip that applies only sometimes doesn't cover the fill.
      code: 'const a = cn("from-foreground-body", on && "bg-clip-text");',
      filename: FILE,
      errors: [{ message: /^"from-foreground-body" paints a fill/ }],
    },
    {
      code: 'const tip = cn("text-surface-muted", isOff && "disabled:text-canvas");',
      filename: FILE,
      errors: [
        {
          message:
            /^"text-surface-muted" colours text with a surface role.*"foreground-inverse" on the inverse surface/,
        },
        { message: /^"disabled:text-canvas" colours text with a surface role/ },
      ],
    },
  ],
});

tester.run("no-raw-z-index", rule("no-raw-z-index"), {
  valid: [
    jsx("relative z-raised hover:z-overlay focus-visible:z-sticky z-auto"),
    // Words and other utilities that end in a z and a number.
    jsx("size-10 translate-z-10 blur-10 [--z-10:1]"),
    // A named layer read through its variable, in either spelling.
    jsx("z-(--z-overlay) hover:z-[var(--z-overlay)] md:z-[var(--z-tour)]"),
    {
      code: 'const a = { zIndex: "var(--z-overlay)" }; const b = { zIndex: layer };',
      filename: FILE,
    },
  ],
  invalid: [
    {
      code: '<div className="z-10 -z-10 hover:z-[21]" />',
      filename: FILE,
      errors: [
        { message: /^"z-10" sets a layer by number\. Use "z-raised"/ },
        { message: /^"-z-10" sets a layer by number\. Use "z-behind"/ },
        {
          message:
            /^"hover:z-\[21\]" sets a layer by number\. Use "z-search-bar"/,
        },
      ],
    },
    {
      // Two layers at one height: the message names both.
      code: 'const corner = cn("fixed z-40", isOpen && "z-[42]");',
      filename: FILE,
      errors: [
        {
          message:
            /Use the layer with that number that names what sits there: "z-header" or "z-prompt"/,
        },
        {
          message:
            /^"z-\[42\]" sets a layer by number\. No layer has this number/,
        },
      ],
    },
    {
      code: 'const a = { zIndex: 50 }; const b = { zIndex: -1 }; const c = { "z-index": "10" };',
      filename: FILE,
      errors: [
        { message: /^"zIndex: 50" sets a layer by number\. Use "z-overlay"/ },
        {
          message:
            /^"zIndex: -1" sets a layer by number\. No layer has this number/,
        },
        { message: /^"zIndex: 10" sets a layer by number\. Use "z-raised"/ },
      ],
    },
  ],
});

tester.run("no-arbitrary-values", rule("no-arbitrary-values"), {
  // An arbitrary colour is no-arbitrary-colour's.
  valid: [jsx("bg-surface"), jsx("bg-[#EFEFF0]")],
  invalid: [
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

tester.run("no-raw-text-size", rule("no-raw-text-size"), {
  valid: [
    jsx("text-body md:text-heading-md text-foreground-base"),
    jsx("text-[var(--accent)] text-base-content text-left"),
  ],
  invalid: [
    {
      ...jsx("text-sm md:text-2xl"),
      errors: [
        {
          message:
            /^"text-sm" is not a step of the type scale\. Use "text-body"/,
        },
        { message: /^"md:text-2xl" is not .* Use "text-heading-md"/ },
      ],
    },
    {
      code: "const styles = cn(`!text-xs ${extra} [&_svg]:text-lg`);",
      filename: FILE,
      errors: [
        { message: /"!text-xs" .* Use "text-caption"/ },
        { message: /"\[&_svg\]:text-lg" .* Use "text-lead"/ },
      ],
    },
    {
      ...jsx("text-[13px] text-[length:var(--size)]"),
      errors: [
        { message: /"text-\[13px\]" .* nearest step \(0\.8125rem\)/ },
        { message: /"text-\[length:var\(--size\)\]" .* a step of the type/ },
      ],
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
  valid: [
    `<Button variant="secondary" />`,
    `const id = "legacy-session";`,
    // Object.prototype names are not variants.
    `if (key === "__proto__" || key === "constructor" || key === "toString") {}`,
  ],
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
  JSON.parse(readFileSync(FIXTURE_BASELINE, "utf8")) as Record<
    string,
    Record<string, number>
  >,
).map(([file, counts]) => [file, counts["no-dark-variant"]] as const)[0];

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

describe("parseFailures", () => {
  it("finds the files oxlint could not parse, which have no rule code", () => {
    expect(
      parseFailures([
        { code: null, filename: "src/Bad.tsx", message: "Unexpected token" },
        { code: "design-system(no-raw-colors)", filename: "src/Ok.tsx" },
      ]),
    ).toEqual([
      { code: null, filename: "src/Bad.tsx", message: "Unexpected token" },
    ]);
  });
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
    // oxlint reports real paths; macOS's tmpdir is under the /var symlink, so
    // the baseline keys must be written from the real path too.
    root = realpathSync(
      mkdtempSync(path.join(tmpdir(), "design-system-lint-")),
    );
    write(
      ".oxlintrc.json",
      JSON.stringify({
        categories: { correctness: "off" },
        jsPlugins: [
          path.join(FRONTEND_ROOT, "scripts/eslint-rules/design-system.mts"),
        ],
        rules: {
          "design-system/no-raw-palette-classes": "error",
          "design-system/no-dark-variant": "error",
          "design-system/baseline-outdated": "warn",
        },
      }),
    );
    write(
      "src/Old.tsx",
      `export const Old = () => <div className="text-zinc-500 dark:bg-surface" />;\n`,
    );
    write(
      "src/Clean.tsx",
      `export const Clean = () => <div className="text-foreground-tertiary" />;\n`,
    );
  });

  afterAll(() => rmSync(root, { recursive: true, force: true }));

  it("passes when every file is at its baseline", () => {
    const { status, output } = lint({
      [key("src/Old.tsx")]: {
        "no-raw-palette-classes": 1,
        "no-dark-variant": 1,
      },
    });
    expect(output).not.toMatch(/design-system/);
    expect(status).toBe(0);
  });

  it("fails on a violation in a file the baseline does not list", () => {
    const { status, output } = lint({});
    expect(output).toMatch(
      /Old\.tsx.*"text-zinc-500" uses the raw Tailwind palette\. Use "text-foreground-tertiary" \(scripts\/ds\/palette-map\.json\)/,
    );
    expect(output).toMatch(/Old\.tsx.*"dark:bg-surface" is a dark: class/);
    expect(status).toBe(1);
  });

  it("fails when one rule's count goes over, and names only that rule", () => {
    const { status, output } = lint({
      [key("src/Old.tsx")]: { "no-raw-palette-classes": 1 },
    });
    expect(output).toMatch(/design-system\(no-dark-variant\)/);
    expect(output).not.toMatch(/design-system\(no-raw-palette-classes\)/);
    expect(status).toBe(1);
  });

  it("does not count suppressed violations against the baseline", () => {
    write(
      "src/Suppressed.tsx",
      [
        `export const A = () => <div className="text-zinc-500" />;`,
        `// oxlint-disable-next-line design-system/no-raw-palette-classes -- brand colour from the partner's spec`,
        `export const B = () => <div className="text-rose-700" />;`,
        `/* eslint-disable design-system/no-raw-palette-classes -- third-party markup */`,
        `export const C = () => <div className="text-sky-500" />;`,
        `/* eslint-enable design-system/no-raw-palette-classes */`,
        `// oxlint-disable-next-line design-system/no-dark-variant -- another rule`,
        `export const D = () => <div className="text-red-500" />;`,
        "",
      ].join("\n"),
    );
    const grandfathered = {
      [key("src/Old.tsx")]: {
        "no-raw-palette-classes": 1,
        "no-dark-variant": 1,
      },
    };
    const atBaseline = lint({
      ...grandfathered,
      [key("src/Suppressed.tsx")]: { "no-raw-palette-classes": 2 },
    });
    const overBaseline = lint({
      ...grandfathered,
      [key("src/Suppressed.tsx")]: { "no-raw-palette-classes": 1 },
    });
    rmSync(path.join(root, "src/Suppressed.tsx"));
    expect(atBaseline.output).not.toMatch(/design-system/);
    expect(atBaseline.status).toBe(0);
    // Only the two unsuppressed classes are reported, and counted.
    expect(overBaseline.output).toMatch(/"text-zinc-500".*now has 2/);
    expect(overBaseline.output).toMatch(/"text-red-500".*now has 2/);
    expect(overBaseline.output).not.toMatch(/text-rose-700|text-sky-500/);
    expect(overBaseline.status).toBe(1);
  });

  it("keeps a rule disabled when an enable names only another rule", () => {
    write(
      "src/Block.tsx",
      [
        `export const Z = () => <div className="text-sky-500" />;`,
        `/* oxlint-disable design-system/no-dark-variant, design-system/no-raw-palette-classes -- vendored markup */`,
        `export const A = () => <div className="dark:block" />;`,
        `/* oxlint-enable design-system/no-dark-variant */`,
        `export const B = () => <div className="text-zinc-500 dark:hidden" />;`,
        "",
      ].join("\n"),
    );
    const { status, output } = lint({
      [key("src/Old.tsx")]: {
        "no-raw-palette-classes": 1,
        "no-dark-variant": 1,
      },
      // The grandfathered raw colour is text-sky-500, outside the block;
      // text-zinc-500 is still disabled and must not push the count over.
      [key("src/Block.tsx")]: {
        "no-raw-palette-classes": 1,
        "no-dark-variant": 1,
      },
    });
    rmSync(path.join(root, "src/Block.tsx"));
    expect(output).not.toMatch(/design-system/);
    expect(status).toBe(0);
  });

  it("tracks nested disable/enable blocks for different rules separately", () => {
    write(
      "src/Nested.tsx",
      [
        `/* oxlint-disable design-system/no-raw-palette-classes -- outer: vendored markup */`,
        `export const A = () => <div className="text-zinc-500" />;`,
        `/* oxlint-disable design-system/no-dark-variant -- inner: vendored theme */`,
        `export const B = () => <div className="text-rose-700 dark:block" />;`,
        `/* oxlint-enable design-system/no-dark-variant */`,
        // The inner block closed first: dark: is back on, raw colours still off.
        `export const C = () => <div className="text-sky-500 dark:hidden" />;`,
        `/* oxlint-enable design-system/no-raw-palette-classes */`,
        `export const D = () => <div className="text-red-500 dark:flex" />;`,
        "",
      ].join("\n"),
    );
    const grandfathered = {
      [key("src/Old.tsx")]: {
        "no-raw-palette-classes": 1,
        "no-dark-variant": 1,
      },
    };
    const atBaseline = lint({
      ...grandfathered,
      [key("src/Nested.tsx")]: {
        "no-raw-palette-classes": 1,
        "no-dark-variant": 2,
      },
    });
    const overBaseline = lint(grandfathered);
    rmSync(path.join(root, "src/Nested.tsx"));
    expect(atBaseline.output).not.toMatch(/design-system/);
    expect(atBaseline.status).toBe(0);
    expect(overBaseline.output).toMatch(/Nested\.tsx.*"text-red-500"/);
    expect(overBaseline.output).toMatch(/Nested\.tsx.*"dark:hidden"/);
    expect(overBaseline.output).toMatch(/Nested\.tsx.*"dark:flex"/);
    expect(overBaseline.output).not.toMatch(
      /text-zinc-500|text-rose-700|text-sky-500|dark:block/,
    );
    expect(overBaseline.status).toBe(1);
  });

  it("tracks overlapping disable/enable blocks for different rules separately", () => {
    write(
      "src/Overlap.tsx",
      [
        `/* oxlint-disable design-system/no-raw-palette-classes -- vendored markup */`,
        `export const A = () => <div className="text-zinc-500" />;`,
        `/* oxlint-disable design-system/no-dark-variant -- vendored theme */`,
        `export const B = () => <div className="text-rose-700 dark:block" />;`,
        // The outer block closes first: raw colours are back on, dark: still off.
        `/* oxlint-enable design-system/no-raw-palette-classes */`,
        `export const C = () => <div className="text-sky-500 dark:hidden" />;`,
        `/* oxlint-enable design-system/no-dark-variant */`,
        `export const D = () => <div className="text-red-500 dark:flex" />;`,
        "",
      ].join("\n"),
    );
    const { status, output } = lint({
      [key("src/Old.tsx")]: {
        "no-raw-palette-classes": 1,
        "no-dark-variant": 1,
      },
    });
    rmSync(path.join(root, "src/Overlap.tsx"));
    expect(output).toMatch(/Overlap\.tsx.*"text-sky-500"/);
    expect(output).toMatch(/Overlap\.tsx.*"text-red-500"/);
    expect(output).toMatch(/Overlap\.tsx.*"dark:flex"/);
    expect(output).not.toMatch(
      /text-zinc-500|text-rose-700|dark:block|dark:hidden/,
    );
    expect(status).toBe(1);
  });

  it("passes with a warning when a file has fewer than its baseline", () => {
    const { status, output } = lint({
      [key("src/Old.tsx")]: {
        "no-raw-palette-classes": 3,
        "no-dark-variant": 1,
      },
    });
    expect(output).toMatch(
      /Old\.tsx.*no-raw-palette-classes 3 -> 1.*pnpm lint:ds-baseline.*design-system\(baseline-outdated\)/,
    );
    expect(status).toBe(0);
  });
});
