// @vitest-environment node
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import {
  compare,
  countViolations,
  isScanned,
  scan,
  serialize,
  type Baseline,
} from "./ds-guard.mjs";

describe("countViolations", () => {
  it("counts raw palette classes with variants, opacity and important", () => {
    const source = `
      <div className="bg-zinc-200 hover:text-neutral-500 border-t-red-300/50 !ring-blue-600 from-slate-50 to-gray-950" />
    `;
    expect(countViolations(source)).toEqual({ "palette-color": 6 });
  });

  it("ignores colour roles, black and white, and palette names inside longer words", () => {
    const source = `
      <div className="bg-surface text-foreground-secondary border-border-base bg-white text-black" />
      const label = "my-bg-red-500"; const x = "text-red-5000"; const step = "bg-red-450";
    `;
    expect(countViolations(source)).toEqual({});
  });

  it("counts arbitrary colour values but not other arbitrary values", () => {
    const source = `
      <div className="bg-[#fff] text-[rgb(0,0,0)] border-[hsl(0_0%_50%)] shadow-[0_1px_2px_rgba(0,0,0,0.1)] bg-[color:var(--x)] fill-[oklch(0.5_0.1_200)]" />
      <div className="w-[12px] text-[14px] bg-[url('/a.png')] text-[var(--x)] shadow-[0_1px_2px_var(--shadow)]" />
    `;
    expect(countViolations(source)).toEqual({ "arbitrary-color": 6 });
  });

  it("counts dark: classes but not object keys or other variants", () => {
    const source = `
      <div className="dark:bg-surface hover:dark:text-white dark:[&_p]:hidden" />
      const theme = { dark: "x" }; const other = "data-[theme=dark]:hidden";
    `;
    expect(countViolations(source)).toEqual({ "dark-variant": 3 });
  });

  it("counts imports from legacy paths in every import form", () => {
    const source = `
      import { Sidebar } from "@/components/__legacy__/Sidebar";
      import type {
        Status,
      } from "../__legacy__/Status";
      export { X } from "./legacy/X";
      import { LegacyButton } from "./LegacyButton";
      const Lazy = dynamic(() => import("@/components/__legacy__/SortDropdown"));
      import "./legacy/styles.css";
      import { readLegacyCookie } from "./legacy-cookies";
      import { legacyCategory } from "./legacyCategory";
      import type { LegacyButtonProps } from "./helpers";
    `;
    expect(countViolations(source)).toEqual({ "legacy-import": 6 });
  });

  it("counts legacy Button variants and sizes", () => {
    const source = `
      <Button variant="legacy-outline" size="legacy-sm" />
      <Button variant={'legacy-ghost'} />
      const name = "legacy-session-token";
    `;
    expect(countViolations(source)).toEqual({ "legacy-button-variant": 3 });
  });
});

describe("isScanned", () => {
  it("scans source, styles and stories but not tests or generated code", () => {
    expect(isScanned("src/components/atoms/Button/Button.tsx")).toBe(true);
    expect(isScanned("src/app/globals.css")).toBe(true);
    expect(isScanned("src/components/atoms/Button/Button.stories.tsx")).toBe(
      true,
    );
    expect(isScanned("src/components/atoms/Button/Button.test.tsx")).toBe(
      false,
    );
    expect(isScanned("src/app/(platform)/library/__tests__/main.tsx")).toBe(
      false,
    );
    expect(isScanned("src/app/api/__generated__/models/agent.ts")).toBe(false);
    expect(isScanned("src/tests/integrations/test-utils.tsx")).toBe(false);
    expect(isScanned("src/playwright/pages/login.page.ts")).toBe(false);
    expect(isScanned("src/types/global.d.ts")).toBe(false);
    expect(isScanned("public/logo.svg")).toBe(false);
  });
});

describe("ratchet against a baseline", () => {
  let root: string;

  function write(file: string, content: string) {
    const full = path.join(root, file);
    mkdirSync(path.dirname(full), { recursive: true });
    writeFileSync(full, content);
  }

  beforeEach(() => {
    root = mkdtempSync(path.join(tmpdir(), "ds-guard-"));
    write("src/Old.tsx", `<div className="bg-zinc-200 dark:bg-zinc-800" />`);
    write("src/Clean.tsx", `<div className="bg-surface" />`);
  });

  afterEach(() => {
    rmSync(root, { recursive: true, force: true });
  });

  it("passes when nothing changed, and only lists files with violations", () => {
    const baseline = scan(root);
    expect(baseline).toEqual({
      "src/Old.tsx": { "palette-color": 2, "dark-variant": 1 },
    });
    expect(compare(baseline, scan(root))).toEqual({
      regressions: [],
      improvements: [],
    });
  });

  it("fails when a file's count of one rule rises", () => {
    const baseline = scan(root);
    write(
      "src/Old.tsx",
      `<div className="bg-zinc-200 text-[#333] dark:bg-zinc-800" />`,
    );
    expect(compare(baseline, scan(root)).regressions).toEqual([
      { file: "src/Old.tsx", rule: "arbitrary-color", allowed: 0, found: 1 },
    ]);
  });

  it("fails when trading one kind of violation for another", () => {
    const baseline = scan(root);
    write("src/Old.tsx", `<div className="bg-zinc-200 bg-[#333]" />`);
    const { regressions, improvements } = compare(baseline, scan(root));
    expect(regressions).toEqual([
      { file: "src/Old.tsx", rule: "arbitrary-color", allowed: 0, found: 1 },
    ]);
    expect(improvements).toEqual([
      { file: "src/Old.tsx", rule: "palette-color", allowed: 2, found: 1 },
      { file: "src/Old.tsx", rule: "dark-variant", allowed: 1, found: 0 },
    ]);
  });

  it("fails when a new or clean file gains a violation", () => {
    const baseline = scan(root);
    write("src/New.tsx", `import { X } from "@/components/__legacy__/X";`);
    write("src/Clean.tsx", `<Button variant="legacy-default" />`);
    expect(compare(baseline, scan(root)).regressions).toEqual([
      {
        file: "src/Clean.tsx",
        rule: "legacy-button-variant",
        allowed: 0,
        found: 1,
      },
      { file: "src/New.tsx", rule: "legacy-import", allowed: 0, found: 1 },
    ]);
  });

  it("passes when counts go down or a file is deleted, and reports them", () => {
    write("src/Gone.tsx", `<div className="text-neutral-500" />`);
    const baseline = scan(root);
    write("src/Old.tsx", `<div className="bg-surface dark:bg-zinc-800" />`);
    rmSync(path.join(root, "src/Gone.tsx"));
    const { regressions, improvements } = compare(baseline, scan(root));
    expect(regressions).toEqual([]);
    expect(improvements).toEqual([
      { file: "src/Gone.tsx", rule: "palette-color", allowed: 1, found: 0 },
      { file: "src/Old.tsx", rule: "palette-color", allowed: 2, found: 1 },
    ]);
  });

  it("serializes the baseline as stable, prettier-formatted JSON", () => {
    const baseline: Baseline = scan(root);
    expect(serialize(baseline)).toBe(
      `{\n  "src/Old.tsx": {\n    "palette-color": 2,\n    "dark-variant": 1\n  }\n}\n`,
    );
  });
});
