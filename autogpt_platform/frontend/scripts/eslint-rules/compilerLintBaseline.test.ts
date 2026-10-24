// @vitest-environment node
import { spawnSync } from "node:child_process";
import { existsSync, readdirSync, readFileSync } from "node:fs";
import path from "node:path";
import { ESLint, type Linter } from "eslint";
import { beforeAll, describe, expect, it } from "vitest";
import { reactCompilerSources } from "../../next.config.mjs";
import { MANUAL_MEMO_BASELINE_FILE } from "./plugin.mjs";

// The compiler-era lint rules are errors where the React Compiler runs and
// warnings elsewhere. Files that already broke a rule are grandfathered at
// "warn" for that one rule (.eslintrc.json, .oxlintrc.json). These tests keep
// both configs pointed at the folders next.config.mjs compiles, and fail when
// a grandfathered file no longer breaks its rule, so the lists only shrink.

const FRONTEND_ROOT = path.resolve(__dirname, "../..");

const COMPILER_RULES = [
  "exhaustive-deps",
  "component-hook-factories",
  "config",
  "error-boundaries",
  "gating",
  "globals",
  "immutability",
  "incompatible-library",
  "preserve-manual-memoization",
  "purity",
  "refs",
  "set-state-in-effect",
  "set-state-in-render",
  "static-components",
  "unsupported-syntax",
  "use-memo",
].map((rule) => `react-hooks/${rule}`);

const eslint = new ESLint({ cwd: FRONTEND_ROOT });

function isTestOrStory(file: string) {
  return (
    /\.(test|spec|stories)\.(js|jsx|ts|tsx)$/.test(file) ||
    file.split("/").includes("__tests__")
  );
}

function isCompiled(file: string) {
  return (
    reactCompilerSources.some((source) => `/${file}`.includes(source)) &&
    !isTestOrStory(file)
  );
}

function listSourceFiles() {
  return readdirSync(path.join(FRONTEND_ROOT, "src"), { recursive: true })
    .map((file) => `src/${String(file).split(path.sep).join("/")}`)
    .filter(
      (file) =>
        /\.(js|jsx|ts|tsx)$/.test(file) &&
        !file.endsWith(".d.ts") &&
        !file.startsWith("src/app/api/__generated__/"),
    )
    .sort();
}

function severity(entry: Linter.RuleEntry | undefined) {
  const level = Array.isArray(entry) ? entry[0] : entry;
  if (level === 2 || level === "error") return "error";
  if (level === 1 || level === "warn") return "warn";
  return "off";
}

function compilerRuleLevels(config: Linter.Config) {
  return COMPILER_RULES.map((rule) => ({
    rule,
    level: severity(config.rules?.[rule]),
  }));
}

describe("ESLint compiler rules", () => {
  const files = listSourceFiles();
  let levels: {
    file: string;
    levels: { rule: string; level: string }[];
  }[];

  beforeAll(async () => {
    levels = await Promise.all(
      files.map(async (file) => ({
        file,
        levels: compilerRuleLevels(await eslint.calculateConfigForFile(file)),
      })),
    );
  }, 120_000);

  it("are errors exactly in the folders the React Compiler compiles", () => {
    const wrong = levels.flatMap(({ file, levels }) => {
      const expected = isCompiled(file) ? "error" : "warn";
      const level = levels.find(({ rule }) => rule === "react-hooks/use-memo");
      return level?.level === expected ? [] : [`${file}: ${level?.level}`];
    });

    expect(wrong).toEqual([]);
  });

  it("keep every compiler rule on, so none can be dropped quietly", () => {
    // Inside the compiled folders a rule may also be "warn" for a
    // grandfathered file; the next test proves each of those reproduces.
    const wrong = levels.flatMap(({ file, levels }) => {
      const allowed = isCompiled(file) ? ["error", "warn"] : ["warn"];
      return levels
        .filter(({ level }) => !allowed.includes(level))
        .map(({ rule, level }) => `${file}: ${rule} is ${level}`);
    });

    expect(wrong).toEqual([]);
  });

  it("still break every rule they are grandfathered for", async () => {
    const grandfathered = levels.flatMap(({ file, levels }) =>
      isCompiled(file)
        ? levels
            .filter(({ level }) => level !== "error")
            .map(({ rule }) => ({ file, rule }))
        : [],
    );
    expect(grandfathered.length).toBeGreaterThan(0);

    const results = await eslint.lintFiles([
      ...new Set(grandfathered.map(({ file }) => file)),
    ]);
    const reported = new Set(
      results.flatMap((result) =>
        result.messages.map(
          (message) =>
            `${path.relative(FRONTEND_ROOT, result.filePath).split(path.sep).join("/")} ${message.ruleId}`,
        ),
      ),
    );
    const stale = grandfathered
      .map(({ file, rule }) => `${file} ${rule}`)
      .filter((key) => !reported.has(key));

    expect(
      stale,
      "These files no longer break the rule; remove them from its grandfathered override in .eslintrc.json.",
    ).toEqual([]);
  }, 120_000);
});

interface OxlintOverride {
  files: string[];
  excludeFiles?: string[];
  rules: Record<string, string>;
}

const oxlintConfig = JSON.parse(
  readFileSync(path.join(FRONTEND_ROOT, ".oxlintrc.json"), "utf8"),
) as { overrides: OxlintOverride[] };

describe("oxlint compiler rules", () => {
  it("are errors in the folders the React Compiler compiles", () => {
    const compiled = oxlintConfig.overrides.find(
      (override) => override.rules["autogpt/no-manual-memo"] === "error",
    );

    expect(compiled?.files).toEqual(
      reactCompilerSources.map((source) => `${source.slice(1)}**`),
    );
    expect(compiled?.rules["autogpt/no-hookless-use-function"]).toBe("error");
  });

  it("still flag every grandfathered file", () => {
    const grandfathered = oxlintConfig.overrides.flatMap((override) =>
      override.rules["autogpt/no-hookless-use-function"] === "warn"
        ? override.files
        : [],
    );
    expect(grandfathered.length).toBeGreaterThan(0);

    const { stdout } = spawnSync(
      path.join(FRONTEND_ROOT, "node_modules/.bin/oxlint"),
      ["--format", "json", ...grandfathered],
      { cwd: FRONTEND_ROOT, encoding: "utf8" },
    );
    const { diagnostics } = JSON.parse(stdout) as {
      diagnostics: { code: string; filename: string }[];
    };
    const stale = grandfathered.filter(
      (file) =>
        !diagnostics.some(
          (diagnostic) =>
            diagnostic.filename === file &&
            diagnostic.code === "autogpt(no-hookless-use-function)",
        ),
    );

    expect(
      stale,
      "These files no longer break no-hookless-use-function; remove them from its grandfathered override in .oxlintrc.json.",
    ).toEqual([]);
  });

  it("keep a manual-memo baseline only for compiled files that exist", () => {
    const baseline = JSON.parse(
      readFileSync(MANUAL_MEMO_BASELINE_FILE, "utf8"),
    ) as Record<string, number>;
    const wrong = Object.entries(baseline).filter(
      ([file, count]) =>
        !existsSync(path.join(FRONTEND_ROOT, file)) ||
        !isCompiled(file) ||
        !(count > 0),
    );

    expect(wrong).toEqual([]);
  });
});
