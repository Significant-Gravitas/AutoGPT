// @vitest-environment node
import { existsSync, readdirSync, readFileSync } from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";
import tailwindConfig from "../../../tailwind.config";
import { colors } from "../styles/colors";

// The contract between the design tokens, the Tailwind config and the
// primitives (SECRT-3097). See docs/engineering/design-system.md.

const FRONTEND_ROOT = path.resolve(__dirname, "../../..");
const globalsCss = readFileSync(
  path.join(FRONTEND_ROOT, "src/app/globals.css"),
  "utf8",
);

// Colours declared in every top-level rule whose selector list includes
// `selector`; empty when there is none.
function declaredColors(selector: string) {
  const names = new Set<string>();
  for (const [, selectors, block] of globalsCss.matchAll(
    /^([^\s{}@/*][^{}]*)\{([^{}]*)\}/gm,
  )) {
    if (!selectors.split(",").some((s) => s.trim() === selector)) continue;
    for (const m of block.matchAll(/--color-([a-z-]+):/g)) names.add(m[1]);
  }
  return names;
}

function registeredColors(value: unknown): string[] {
  if (typeof value === "string")
    return [...value.matchAll(/var\(--color-([a-z-]+)\)/g)].map((m) => m[1]);
  if (value && typeof value === "object")
    return Object.values(value).flatMap(registeredColors);
  return [];
}

describe("colour tokens", () => {
  const root = declaredColors(":root");
  const registered = new Set(
    registeredColors(tailwindConfig.theme?.extend?.colors),
  );

  it("registers every colour globals.css declares, and nothing it does not", () => {
    expect([...root].filter((name) => !registered.has(name))).toEqual([]);
    expect([...registered].filter((name) => !root.has(name))).toEqual([]);
  });

  it("only overrides names :root declares in .dark", () => {
    expect(root.size).toBeGreaterThan(0);
    const dark = declaredColors(".dark");
    expect([...dark].filter((name) => !root.has(name))).toEqual([]);
  });

  it("has one dark selector, `.dark`", () => {
    expect(tailwindConfig.darkMode).toBe("class");
    expect(globalsCss).not.toMatch(/\.dark-mode|data-theme=["']?dark/);
  });

  it("keeps the raw palette frozen; new colours are roles in globals.css", () => {
    expect(Object.keys(colors).sort()).toEqual(
      [
        "bgLightGrey",
        "black",
        "green",
        "orange",
        "pink",
        "purple",
        "red",
        "slate",
        "textBlack",
        "textGrey",
        "white",
        "yellow",
        "zinc",
      ].sort(),
    );
  });
});

describe("components.json", () => {
  const config = JSON.parse(
    readFileSync(path.join(FRONTEND_ROOT, "components.json"), "utf8"),
  ) as {
    tailwind: { config: string; css: string };
    aliases: Record<string, string>;
  };

  it("points at files that exist", () => {
    expect(existsSync(path.join(FRONTEND_ROOT, config.tailwind.config))).toBe(
      true,
    );
    expect(existsSync(path.join(FRONTEND_ROOT, config.tailwind.css))).toBe(
      true,
    );
  });

  it("has aliases that resolve", () => {
    for (const alias of Object.values(config.aliases)) {
      const target = path.join(FRONTEND_ROOT, alias.replace(/^@\//, "src/"));
      const found = ["", ".ts", ".tsx"].some((ext) =>
        existsSync(`${target}${ext}`),
      );
      expect(found, alias).toBe(true);
    }
  });
});

// Primitives mark their root element with data-slot so callers and tests can
// target their parts. These predate the rule; fix one and delete it here.
const UNSLOTTED_PRIMITIVES = [
  "src/components/atoms/RelativeTime/RelativeTime.tsx",
  "src/components/molecules/Dialog/Dialog.tsx",
];

function primitiveFiles() {
  return ["atoms", "molecules"].flatMap((level) => {
    const dir = path.join(FRONTEND_ROOT, "src/components", level);
    return readdirSync(dir)
      .map((name) => `src/components/${level}/${name}/${name}.tsx`)
      .filter((file) => existsSync(path.join(FRONTEND_ROOT, file)));
  });
}

describe("primitives", () => {
  it("set data-slot", () => {
    const unslotted = primitiveFiles().filter(
      (file) =>
        !readFileSync(path.join(FRONTEND_ROOT, file), "utf8").includes(
          "data-slot",
        ),
    );
    expect(unslotted).toEqual(UNSLOTTED_PRIMITIVES);
  });
});
