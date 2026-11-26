// @vitest-environment node
import { readdirSync, readFileSync, statSync } from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";

// Every UI transition takes its curve from motion.ts (`ease.out`,
// `ease.emphasized`, ...). This fails on a curve written out in a source file:
// a cubic-bezier array, one of Framer Motion's named easings, or an *EASE*
// array constant. See "How to animate" in docs/engineering/design-system.md.

const FRONTEND = path.resolve(__dirname, "../../../..");
const SKIP_DIRS = new Set([
  "node_modules",
  "__generated__",
  "__tests__",
  "playwright",
  "tests",
  "mocks",
]);
const TOKENS = "src/components/styles/motion.ts";

// The loops and ambient effects keep the curve they were drawn with; they are
// not UI transitions.
const ALLOWED: Record<string, string> = {
  "src/app/(no-navbar)/onboarding/steps/BrainDumpStep/components/AmbientCard.tsx":
    'The idle drift is an endless "easeInOut" loop, not a transition.',
  "src/app/(no-navbar)/onboarding/steps/BrainDumpStep/components/TypingDots.tsx":
    'The typing dots blink on an endless "easeInOut" loop.',
  "src/app/(platform)/copilot/components/EmptySession/components/GreetingLoader.tsx":
    'The loader breathes on an endless "easeInOut" loop.',
  "src/components/contextual/PublishAgentModal/components/ReviewHero.tsx":
    'The celebration rings are a 1.6s "easeOut" pulse that repeats.',
};

const RULES: { name: string; pattern: RegExp }[] = [
  {
    name: "cubic-bezier array",
    pattern:
      /\b\w*ease\w*\s*(?::[^=\n]*)?[:=]\s*\[\s*-?\d*\.?\d+\s*(?:,\s*-?\d*\.?\d+\s*){3}\]/gi,
  },
  { name: "cubicBezier() call", pattern: /\bcubicBezier\s*\(/g },
  {
    name: "Framer Motion easing name",
    pattern:
      /["'`](?:easeIn|easeOut|easeInOut|circIn|circOut|circInOut|backIn|backOut|backInOut|anticipate)["'`]/g,
  },
  {
    name: "*EASE* array constant",
    pattern: /\bconst\s+\w*EASE\w*\s*(?::[^=\n]+)?=\s*\[/g,
  },
];

function walk(dir: string, out: string[] = []) {
  for (const entry of readdirSync(dir)) {
    if (SKIP_DIRS.has(entry)) continue;
    const full = path.join(dir, entry);
    if (statSync(full).isDirectory()) walk(full, out);
    else out.push(path.relative(FRONTEND, full).split(path.sep).join("/"));
  }
  return out;
}

function isSource(file: string) {
  return (
    /\.(ts|tsx)$/.test(file) &&
    !/\.(test|spec|stories)\.tsx?$/.test(file) &&
    file !== TOKENS
  );
}

// Blanks comments out but keeps their newlines, so line numbers still hold.
function withoutComments(text: string) {
  return text
    .replace(/\/\*[\s\S]*?\*\//g, (comment) => comment.replace(/[^\n]/g, " "))
    .replace(/(^|[^:"'`\\])\/\/[^\n]*/g, "$1");
}

function findings(file: string) {
  const text = withoutComments(readFileSync(path.join(FRONTEND, file), "utf8"));
  return RULES.flatMap(({ name, pattern }) =>
    [...text.matchAll(pattern)].map((match) => {
      const line = text.slice(0, match.index).split("\n").length;
      return `${file}:${line} ${name}: ${match[0].replace(/\s+/g, " ")}`;
    }),
  );
}

const SOURCE = walk(path.join(FRONTEND, "src")).filter(isSource);

describe("motion literals", () => {
  it("scans the app's source", () => {
    expect(SOURCE.length).toBeGreaterThan(500);
    expect(SOURCE).toContain("src/components/atoms/FadeIn/FadeIn.tsx");
  });

  it("finds every kind of literal it is meant to catch", () => {
    const sample = [
      "const PANEL_EASE = [0.32, 0.72, 0, 1] as const;",
      "transition={{ ease: [0.16, 1, 0.3, 1] }}",
      'transition={{ ease: "easeOut" }}',
      "const EASE_OUT = cubicBezier(0.4, 0, 0.2, 1);",
    ].join("\n");
    for (const { name, pattern } of RULES)
      expect([...sample.matchAll(pattern)].length, name).toBeGreaterThan(0);
    const allowed = 'transition={{ ease: ease.out }}\n{ ease: "linear" }';
    for (const { name, pattern } of RULES)
      expect([...allowed.matchAll(pattern)], name).toEqual([]);
  });

  it("leaves no easing written out outside motion.ts", () => {
    const found = SOURCE.filter((file) => !(file in ALLOWED)).flatMap(findings);
    expect(found).toEqual([]);
  });

  it("allows only files that still need their exception", () => {
    for (const [file, reason] of Object.entries(ALLOWED)) {
      expect(reason.trim().length, file).toBeGreaterThan(10);
      expect(SOURCE, file).toContain(file);
      expect(
        findings(file).length,
        `${file} no longer needs it`,
      ).toBeGreaterThan(0);
    }
  });
});
