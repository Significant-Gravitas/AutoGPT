import betterTailwind from "eslint-plugin-better-tailwindcss";
import { getDefaultSelectors } from "eslint-plugin-better-tailwindcss/defaults";
import allowlist from "./eslint-allowlist.json" with { type: "json" };

// Design-system lint rules: import boundaries and Tailwind class checks, plus
// the allowlist of existing violations. Kept apart from eslint.config.mjs so
// scripts/*.test.ts can import it without loading the Next.js presets.
// DESIGN.md lists the rules; docs/engineering/frontend-design-system-audit.md
// (Part 8.3 step 1) has the history.

// Build output, generated code and vendored files. eslint.config.mjs ignores
// them, and scripts/eslint-allowlist-regenerate.ts skips them too.
export const LINT_IGNORES = [
  ".next/**",
  "node_modules/**",
  "public/**",
  "coverage/**",
  "storybook-static/**",
  "playwright-report/**",
  "test-results/**",
  "src/app/api/__generated__/**",
  "next-env.d.ts",
];

const HUGEICONS_ONLY =
  'Hugeicons only: import the icon from "@hugeicons/core-free-icons" and render it with the Icon atom.';

// Design-system import boundaries. Each key is one restriction, and the
// allowlist exempts a file from specific keys only, so a file allowlisted for
// `lucide-react` still cannot add a `__legacy__` import.
// Covered by scripts/eslint-design-system-rules.test.ts.
export const IMPORT_RESTRICTIONS = {
  legacy: {
    patterns: [
      {
        regex: "^(@/|(\\.\\./)+)components/__legacy__/",
        message:
          "src/components/__legacy__ is being removed. Use the atoms, molecules and organisms in src/components instead.",
      },
    ],
  },
  ui: {
    patterns: [
      {
        regex: "^(@/|(\\.\\./)+)components/ui/",
        message:
          "src/components/ui is raw shadcn output for the design system to build on. Outside src/components, use the atoms, molecules and organisms instead.",
      },
    ],
  },
  "lucide-react": {
    paths: [{ name: "lucide-react", message: HUGEICONS_ONLY }],
  },
  "@phosphor-icons/react": {
    patterns: [
      { regex: "^@phosphor-icons/react(/|$)", message: HUGEICONS_ONLY },
    ],
  },
  "@radix-ui/react-icons": {
    paths: [{ name: "@radix-ui/react-icons", message: HUGEICONS_ONLY }],
  },
  "@tabler/icons-react": {
    paths: [{ name: "@tabler/icons-react", message: HUGEICONS_ONLY }],
  },
  "react-icons": {
    patterns: [{ regex: "^react-icons(/|$)", message: HUGEICONS_ONLY }],
  },
  "@hugeicons/react": {
    paths: [
      {
        name: "@hugeicons/react",
        allowTypeImports: true,
        message:
          "Render Hugeicons through the Icon atom (src/components/atoms/Icon), which applies the design-system stroke width. Type-only imports such as IconSvgElement are fine.",
      },
    ],
  },
  sonner: {
    paths: [
      {
        name: "sonner",
        message:
          "Show toasts through src/components/molecules/Toast instead of calling sonner directly.",
      },
    ],
  },
};

// Directories a restriction does not apply to by design, as path prefixes
// relative to this file. Nested scopes add up.
export const IMPORT_SCOPES = {
  "src/components/": ["ui"],
  // Kobra registry output, kept as installed so it can be updated. It ships
  // its own Tabler glyphs; the design system's atoms still use Hugeicons.
  "src/components/ui/": ["@tabler/icons-react"],
  "src/components/__legacy__/": ["legacy"],
  "src/components/atoms/Icon/": ["@hugeicons/react"],
  "src/components/molecules/Toast/": ["sonner"],
};

// Existing violations: files exempted per import restriction and per
// Tailwind rule. This is the ratchet, so entries are only ever removed.
// scripts/eslint-allowlist.test.ts fails on an entry that no longer has the
// violation it excuses.
export const ALLOWLIST = allowlist;

// Allowlist entries are literal paths, and Next.js route folders such as
// `[id]` and `(platform)` are glob syntax, so escape them for `files`.
function literalGlob(path) {
  return path.replace(/[*?!+@()[\]{}]/g, "\\$&");
}

// The restriction an import source falls under, or undefined.
export function restrictionFor(source) {
  return Object.entries(IMPORT_RESTRICTIONS).find(
    ([, { paths = [], patterns = [] }]) =>
      paths.some(({ name }) => name === source) ||
      patterns.some(({ regex }) => new RegExp(regex, "iu").test(source)),
  )?.[0];
}

function scopeExemptions(path) {
  return Object.entries(IMPORT_SCOPES)
    .filter(([prefix]) => path.startsWith(prefix))
    .flatMap(([, keys]) => keys);
}

function restrictedImportsRule(exempt) {
  const active = Object.entries(IMPORT_RESTRICTIONS).filter(
    ([key]) => !exempt.includes(key),
  );
  if (active.length === 0) return "off";
  return [
    "error",
    {
      paths: active.flatMap(([, restriction]) => restriction.paths ?? []),
      patterns: active.flatMap(([, restriction]) => restriction.patterns ?? []),
    },
  ];
}

export function importBlocks({ allowlist = true } = {}) {
  const scopeBlocks = Object.keys(IMPORT_SCOPES)
    .sort((a, b) => a.length - b.length)
    .map((prefix) => ({
      name: `design-system/imports:${prefix}`,
      files: [`${prefix}**`],
      rules: {
        "@typescript-eslint/no-restricted-imports": restrictedImportsRule(
          scopeExemptions(prefix),
        ),
      },
    }));

  // One block per distinct set of exemptions, so files allowlisted for the
  // same restrictions share a block.
  const keysByFile = new Map();
  for (const [key, files] of Object.entries(
    allowlist ? ALLOWLIST.imports : {},
  )) {
    for (const file of files) {
      keysByFile.set(file, [...(keysByFile.get(file) ?? []), key]);
    }
  }
  const blocksByExemptions = new Map();
  for (const [file, keys] of keysByFile) {
    const exempt = [...new Set([...scopeExemptions(file), ...keys])].sort();
    const id = exempt.join(",");
    const block = blocksByExemptions.get(id) ?? {
      name: `allowlist/imports:${id}`,
      files: [],
      rules: {
        "@typescript-eslint/no-restricted-imports":
          restrictedImportsRule(exempt),
      },
    };
    block.files.push(literalGlob(file));
    blocksByExemptions.set(id, block);
  }

  return [
    {
      name: "design-system/imports",
      rules: {
        "@typescript-eslint/no-restricted-imports": restrictedImportsRule([]),
      },
    },
    ...scopeBlocks,
    ...blocksByExemptions.values(),
  ];
}

// Classes that are not Tailwind utilities but are legitimate: hooks read by
// third-party libraries, selectors for CSS written outside Tailwind, and the
// plain CSS classes in src/app/globals.css.
export const NON_TAILWIND_CLASSES = [
  "^ph-no-capture$", // PostHog: exclude from autocapture
  "^sentry-(un)?mask$", // Sentry session replay
  "^no(drag|pan|wheel)$", // React Flow interaction guards
  "^react-flow__", // React Flow internals
  "^rjsf-", // react-jsonschema-form hooks
  "^is-(user|assistant)$", // ai-elements message selectors (group-[.is-user])
  "^hide-password-toggle$", // PasswordInput's inline <style>
  "^markdown-output$", // src/app/globals.css
];

// Tailwind default-palette families that are not part of the design system,
// with the palette family each one maps to (see DESIGN.md, "Rules"). The
// families in src/components/styles/colors.ts override Tailwind's, so only
// these fall through to the defaults. `blue`, `sky`, `teal` and `cyan` are
// left out until there is a decision for them.
export const RESTRICTED_PALETTE = {
  gray: "zinc",
  neutral: "zinc",
  stone: "zinc",
  emerald: "green",
  amber: "yellow",
  violet: "purple",
  indigo: "purple",
  rose: "red",
  lime: undefined,
  fuchsia: undefined,
};

// Every utility that takes a colour, after any variants (`hover:`,
// `[&_svg]:`) and an optional `!`, with an optional `/opacity` suffix.
const COLOUR_UTILITIES =
  "bg|text|border(?:-[xytrblse])?|divide|outline|ring(?:-offset)?|shadow|from|via|to|fill|stroke|decoration|accent|caret|placeholder";

function paletteRestriction([family, replacement]) {
  const instead = replacement
    ? `Use \`${replacement}\` at the same step, or a semantic class`
    : "No mapping is decided for it yet; use the nearest palette family or a semantic class";
  return {
    pattern: `(?:^|:)!?(?:${COLOUR_UTILITIES})-${family}-(?:50|[1-9]00|950)(?:/\\S+)?!?$`,
    message: `\`${family}\` is Tailwind's default palette, not a design-system colour. ${instead}. See autogpt_platform/frontend/DESIGN.md.`,
  };
}

// Dark mode comes from the `.dark` block of semantic variables in
// src/app/globals.css, never from per-class overrides.
const DARK_VARIANT_RESTRICTION = {
  pattern: "(?:^|:)dark:",
  message:
    "No `dark:` classes: dark mode will swap the semantic variables in the `.dark` block of src/app/globals.css. Use a semantic class instead. See autogpt_platform/frontend/DESIGN.md.",
};

export function tailwindBlocks({ allowlist = true } = {}) {
  return [
    {
      name: "design-system/tailwind",
      files: ["src/**/*.{ts,tsx}"],
      // Kobra registry files are vendored verbatim (see MIGRATION.md).
      ignores: ["src/components/ui/**"],
      plugins: { "better-tailwindcss": betterTailwind },
      settings: {
        "better-tailwindcss": {
          entryPoint: "src/app/globals.css",
          selectors: [
            ...getDefaultSelectors(),
            // Class maps such as `modalStyles = { content: "..." }` and
            // strings such as `buttonClasses`, which the defaults (exact
            // names only) miss. `*Styles` strings are often raw CSS, so only
            // their object values count.
            {
              kind: "variable",
              name: ".*[sS]tyles?",
              match: [{ type: "objectValues" }],
            },
            {
              kind: "variable",
              name: ".*[cC]lass(?:Name)?(?:s|es)?",
              match: [{ type: "strings" }, { type: "objectValues" }],
            },
          ],
        },
      },
      rules: {
        "better-tailwindcss/no-unknown-classes": [
          "error",
          { ignore: NON_TAILWIND_CLASSES },
        ],
        "better-tailwindcss/no-conflicting-classes": "error",
        "better-tailwindcss/no-restricted-classes": [
          "error",
          {
            restrict: [
              ...Object.entries(RESTRICTED_PALETTE).map(paletteRestriction),
              DARK_VARIANT_RESTRICTION,
            ],
          },
        ],
      },
    },
    ...Object.entries(allowlist ? ALLOWLIST.tailwind : {})
      .filter(([, files]) => files.length > 0)
      .map(([rule, files]) => ({
        name: `allowlist/tailwind:${rule}`,
        files: files.map(literalGlob),
        rules: { [`better-tailwindcss/${rule}`]: "off" },
      })),
  ];
}
