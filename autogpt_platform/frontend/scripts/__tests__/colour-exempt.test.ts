// @vitest-environment node
import { readdirSync, readFileSync, statSync } from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";
import { exemptEntries, isExempt } from "../ds/colour-exempt";
import { buildPaletteMap, PALETTE_MAP } from "../ds/palette-to-role";
import { COLOR_ROLES } from "../../src/components/styles/color-roles";

// One colour system (SECRT-3154): what oxlint cannot check. It lints only
// TypeScript, so the stylesheets are read here, and the exempt list and the
// palette table are kept honest.

const FRONTEND = path.resolve(__dirname, "../..");
const SKIP = new Set(["node_modules", "__generated__"]);

function walk(dir: string, out: string[] = []) {
  for (const entry of readdirSync(dir)) {
    if (SKIP.has(entry)) continue;
    const full = path.join(dir, entry);
    if (statSync(full).isDirectory()) walk(full, out);
    else out.push(path.relative(FRONTEND, full).split(path.sep).join("/"));
  }
  return out;
}

const SOURCE = walk(path.join(FRONTEND, "src"));
const read = (file: string) => readFileSync(path.join(FRONTEND, file), "utf8");

function withoutComments(css: string) {
  return css.replace(/\/\*[\s\S]*?\*\//g, "");
}

const HEX =
  /(?<![\w&#/-])#(?:[0-9a-f]{8}|[0-9a-f]{6}|[0-9a-f]{3,4})(?![0-9a-z-])/gi;

describe("ds-colour-exempt.json", () => {
  const entries = exemptEntries();

  it("gives every entry a reason and a file it still matches", () => {
    for (const entry of entries) {
      expect(entry.reason.trim().length, entry.files).toBeGreaterThan(10);
      expect(
        SOURCE.some((file) => entry.pattern.test(file)),
        `${entry.files} matches no file`,
      ).toBe(true);
    }
  });

  it("allows only values the file still uses", () => {
    for (const entry of entries.filter((e) => e.allow)) {
      const text = SOURCE.filter((file) => entry.pattern.test(file))
        .map(read)
        .join("\n")
        .toLowerCase();
      for (const value of entry.allow ?? [])
        expect(text, `${entry.files}: ${value}`).toContain(value.toLowerCase());
    }
  });
});

describe("stylesheets", () => {
  it("write no hex colour outside the token file and the exempt list", () => {
    const found = SOURCE.filter((file) => file.endsWith(".css")).flatMap(
      (file) =>
        (withoutComments(read(file)).match(HEX) ?? [])
          .filter((hex) => !isExempt(file, hex))
          .map((hex) => `${file}: ${hex}`),
    );
    expect(found).toEqual([]);
  });

  it("keep every colour value in tokens/theme.css", () => {
    const globals = withoutComments(read("src/app/globals.css"));
    expect(globals).not.toMatch(/--color-[a-z-]+:\s*\d+ \d+ \d+;/);
    expect(globals).not.toMatch(/rgba?\(\s*\d/);
  });
});

describe("suppressions", () => {
  it("never switch off the colour rules: an exemption goes in the list", () => {
    const found = SOURCE.filter((file) => /\.(tsx?|mdx)$/.test(file)).filter(
      (file) =>
        /(?:eslint|oxlint)-disable[^\n]*design-system\/no-(?:alias|palette|hex)-colours/.test(
          read(file),
        ),
    );
    expect(found).toEqual([]);
  });
});

describe("palette-map.json", () => {
  const roles = new Set<string>(COLOR_ROLES);

  it("maps every step to roles that exist", () => {
    for (const [step, row] of Object.entries(PALETTE_MAP.steps)) {
      for (const target of [row.text, row.fill, row.line]) {
        if (!target) continue;
        expect(roles.has(target.split("/")[0]), `${step} -> ${target}`).toBe(
          true,
        );
      }
      expect(["none", "slight", "bigger"]).toContain(row.change);
    }
  });

  it("maps whole-class overrides to role classes", () => {
    const map = buildPaletteMap();
    for (const from of Object.keys(PALETTE_MAP.classes)) {
      const to = (map.get(from) ?? "").split("/")[0];
      expect(
        [...roles].some((name) => to.endsWith(`-${name}`)),
        `${from} -> ${to}`,
      ).toBe(true);
    }
  });
});
