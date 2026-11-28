import { readFileSync } from "node:fs";
import path from "node:path";

// Reading the token files, for the tests that check a look or a theme
// against the defaults (canary-look, theme-contrast).

export const FRONTEND_ROOT = path.resolve(__dirname, "../../..");

export const DEFAULT_TOKEN_FILES = [
  "src/app/tokens/theme.css",
  "src/app/tokens/typography.css",
  "src/app/tokens/shape.css",
  "src/app/tokens/motion.css",
  "src/components/styles/fonts.css",
];

export function readCss(file: string) {
  return readFileSync(path.join(FRONTEND_ROOT, file), "utf8").replace(
    /\/\*[\s\S]*?\*\//g,
    "",
  );
}

// The custom properties set in every block `selector` matches, its first
// group being the block's body.
export function declarations(css: string, selector: RegExp) {
  return new Map(
    [...css.matchAll(selector)].flatMap(([, body]) =>
      [...body.matchAll(/(--[a-z0-9-]+):\s*([^;]+);/g)].map(
        ([, name, value]) => [name, value.replace(/\s+/g, " ").trim()] as const,
      ),
    ),
  );
}

// Every token the default look and light theme declare on `:root`.
export function defaultTokens() {
  return new Map(
    DEFAULT_TOKEN_FILES.flatMap((file) => [
      ...declarations(readCss(file), /^:root\s*\{([^}]*)\}/gm),
    ]),
  );
}

export const TRIPLET = /^\d{1,3} \d{1,3} \d{1,3}$/;
