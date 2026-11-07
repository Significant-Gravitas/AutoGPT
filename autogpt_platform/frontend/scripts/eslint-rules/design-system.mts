import { readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { plugin as shadcn } from "@shadcn/lint";
import type { Rule } from "eslint";
import {
  buildReplacementMap,
  buildSizeMap,
  rewriteClassList,
} from "../codemod-tokens.ts";
import { ALIAS_ROLES } from "../ds/alias-to-role.ts";
import { isExempt, isWholeFileExempt } from "../ds/colour-exempt.ts";
import {
  COLOUR_UTILITIES,
  colourClassPattern,
} from "../ds/colour-utilities.ts";
import { buildPaletteMap, PALETTE_VALUES } from "../ds/palette-to-role.ts";
import { RAW_TEXT_SIZE, rawTextSizeAdvice } from "../ds/text-size-scale.ts";

// Design-system lint rules (SECRT-3097), loaded by oxlint as a JS plugin
// (.oxlintrc.json) and run by `pnpm lint`. See
// docs/engineering/design-system.md for the rules themselves.
//
// Every rule here is a ratchet. design-system-baseline.json holds, per file and
// rule, how many violations the file had when the rule was turned on. A file
// at or under its count passes; one over it fails with every violation listed;
// a file missing from the baseline must have none. `pnpm lint:ds-baseline`
// rewrites the baseline, and `design-system/baseline-outdated` fails when a
// file has fewer violations than the baseline allows.
//
// The @shadcn/lint rules are wrapped rather than enabled directly, so that
// they share the baseline and so their raw-colour and arbitrary-value
// messages can name the design token with the same value.

// components.json keeps tailwind.css at globals.css, the stylesheet the app
// loads and the shadcn CLI writes to. It has no `@import "tailwindcss"`
// (Tailwind 3 compat), so @shadcn/lint reads the classes Tailwind knows from
// tailwind-grammar.css instead and says so on every run; that is the setup,
// not a fault, so the one notice is dropped.
const EXPECTED_NOTICE =
  "[@shadcn/lint] components.json sets tailwind.css to src/app/globals.css, which does not import Tailwind. Using src/app/tailwind-grammar.css";
const warn = console.warn;
console.warn = (...args: unknown[]) => {
  if (typeof args[0] === "string" && args[0].startsWith(EXPECTED_NOTICE))
    return;
  warn(...args);
};

type Node = Rule.Node;
type Report = Rule.ReportDescriptor & {
  message?: string;
  messageId?: string;
  data?: Record<string, unknown>;
};
type Baseline = Record<string, Record<string, number>>;

const FRONTEND_ROOT = fileURLToPath(new URL("../..", import.meta.url));
const PLUGIN_NAME = "design-system";
const DOCS = "See docs/engineering/design-system.md.";

export const DESIGN_SYSTEM_BASELINE_FILE = fileURLToPath(
  new URL("./design-system-baseline.json", import.meta.url),
);
// Tests point the rules at a baseline of their own.
const BASELINE_FILE =
  process.env.DESIGN_SYSTEM_BASELINE_FILE ?? DESIGN_SYSTEM_BASELINE_FILE;

// `pnpm lint:ds-baseline` sets this to count every violation.
const BASELINE_OFF = process.env.DESIGN_SYSTEM_BASELINE === "off";
let baseline: Baseline | null = null;

function allowedCount(file: string, rule: string) {
  if (BASELINE_OFF) return 0;
  baseline ??= JSON.parse(readFileSync(BASELINE_FILE, "utf8")) as Baseline;
  return baseline[file]?.[rule] ?? 0;
}

// Per file, how many violations each ratchet rule found. Filled at
// Program:exit and read by baseline-outdated's `after` hook, which oxlint
// runs once the whole file has been walked.
const countsByFile = new Map<string, Map<string, number>>();

function relativeToFrontend(filename: string) {
  return path.relative(FRONTEND_ROOT, filename).split(path.sep).join("/");
}

function renderMessage(report: Report, messages: Record<string, string>) {
  if (report.message) return report.message;
  const template = messages[report.messageId ?? ""] ?? report.messageId ?? "";
  return template.replace(/\{\{\s*(\w+)\s*\}\}/g, (_, key: string) =>
    String(report.data?.[key] ?? ""),
  );
}

type Rewrite = (message: string, report: Report) => string;

// eslint-/oxlint-disable comments, so that a suppressed report never counts
// against the baseline: the baseline is written from oxlint's output, which
// already leaves suppressed reports out.
interface Suppressions {
  // Per line, the rule lists of the -line and -next-line directives on it
  // (null: every rule).
  lines: Map<number, (string[] | null)[]>;
  // disable/enable block directives in source order.
  blocks: {
    kind: "disable" | "enable";
    line: number;
    rules: string[] | null;
  }[];
}

const DIRECTIVE_PARTS =
  /^\s*(?:eslint|oxlint)-(disable-next-line|disable-line|disable|enable)(?:\s+(.*))?$/s;

function parseSuppressions(sourceCode: Rule.RuleContext["sourceCode"]) {
  const found: Suppressions = { lines: new Map(), blocks: [] };
  const addLine = (line: number, rules: string[] | null) =>
    found.lines.set(line, [...(found.lines.get(line) ?? []), rules]);
  for (const comment of sourceCode.getAllComments()) {
    const directive = comment.value.split(/\s--\s/)[0];
    const match = DIRECTIVE_PARTS.exec(directive);
    if (!match || !comment.loc) continue;
    const [, kind, list = ""] = match;
    const names = list
      .split(",")
      .map((rule) => rule.trim())
      .filter(Boolean);
    const rules = names.length > 0 ? names : null;
    const { start, end } = comment.loc;
    if (kind === "disable-next-line") addLine(end.line + 1, rules);
    else if (kind === "disable-line")
      for (let line = start.line; line <= end.line; line++)
        addLine(line, rules);
    else if (kind === "disable")
      found.blocks.push({ kind, line: end.line, rules });
    else found.blocks.push({ kind: "enable", line: start.line, rules });
  }
  return found;
}

const suppressionsBySource = new WeakMap<object, Suppressions>();

function reportLine(report: Report) {
  const { loc, node } = report as {
    loc?: { line?: number; start?: { line: number } };
    node?: Node;
  };
  return loc?.start?.line ?? loc?.line ?? node?.loc?.start.line ?? 0;
}

function isSuppressed(context: Rule.RuleContext, rule: string, report: Report) {
  const sourceCode = context.sourceCode;
  let suppressions = suppressionsBySource.get(sourceCode);
  if (!suppressions) {
    suppressions = parseSuppressions(sourceCode);
    suppressionsBySource.set(sourceCode, suppressions);
  }
  const id = `${PLUGIN_NAME}/${rule}`;
  const covers = (rules: string[] | null) => !rules || rules.includes(id);
  const line = reportLine(report);
  if (suppressions.lines.get(line)?.some(covers)) return true;
  // Walk the block directives for this rule alone, so that enabling one rule
  // leaves the others a directive disabled still off.
  let disabledFrom: number | null = null;
  for (const block of suppressions.blocks) {
    if (!covers(block.rules)) continue;
    if (block.kind === "disable") {
      disabledFrom ??= block.line;
      continue;
    }
    if (disabledFrom !== null && line > disabledFrom && line < block.line)
      return true;
    disabledFrom = null;
  }
  return disabledFrom !== null && line > disabledFrom;
}

/**
 * Collects a rule's reports for one file and lets them through only when the
 * file has more than design-system-baseline.json allows.
 */
function ratchet(
  name: string,
  inner: Rule.RuleModule,
  rewrite?: Rewrite,
  keep: (report: Report, file: string) => boolean = () => true,
): Rule.RuleModule {
  const messages = (inner.meta?.messages ?? {}) as Record<string, string>;
  return {
    meta: { ...inner.meta },
    create(context) {
      const reports: Report[] = [];
      const file = relativeToFrontend(context.filename);
      const target = context as unknown as Record<PropertyKey, unknown>;
      const collecting = new Proxy(
        {},
        {
          get(_, key) {
            if (key === "report")
              return (report: Report) => {
                if (keep(report, file)) reports.push(report);
              };
            const value = Reflect.get(target, key, target);
            return typeof value === "function" ? value.bind(target) : value;
          },
          has: (_, key) => key in target,
        },
      ) as Rule.RuleContext;
      const visitor = inner.create(collecting) as Record<
        string,
        ((node: Node) => void) | undefined
      >;
      const innerExit = visitor["Program:exit"];
      return {
        ...visitor,
        "Program:exit"(program: Node) {
          innerExit?.(program);
          const live = reports.filter(
            (report) => !isSuppressed(context, name, report),
          ).length;
          const counts = countsByFile.get(file) ?? new Map<string, number>();
          counts.set(name, live);
          countsByFile.set(file, counts);
          const allowed = allowedCount(file, name);
          if (live <= allowed) return;
          // oxlint drops the suppressed ones from what is reported.
          const over =
            allowed > 0
              ? ` This file is allowed ${allowed} existing ${PLUGIN_NAME}/${name} violation(s) (scripts/eslint-rules/design-system-baseline.json) and now has ${live}.`
              : "";
          for (const report of reports) {
            let message = renderMessage(report, messages);
            if (rewrite) message = rewrite(message, report);
            const { messageId: _id, data: _data, ...rest } = report;
            context.report({
              ...rest,
              message: `${message}${over} ${DOCS}`,
            } as Rule.ReportDescriptor);
          }
        },
      } as Rule.RuleListener;
    },
  };
}

// ---------------------------------------------------------------------------
// Colour roles, read from src/app/tokens/theme.css so the advice never drifts.

const BUILDER_ROLES =
  /^(?:builder-canvas|builder-grid|node-surface|node-border|node-selected|edge|handle)$/;

interface ColorRole {
  name: string;
  hex: string;
  palette: string | null;
}

function readColorRoles(): ColorRole[] {
  const css = readFileSync(
    path.join(FRONTEND_ROOT, "src/app/tokens/theme.css"),
    "utf8",
  );
  const start = css.indexOf(":root {");
  const root = css.slice(start, css.indexOf("\n}", start));
  const roles: ColorRole[] = [];
  const line =
    /--color-([a-z-]+):\s*(\d+)\s+(\d+)\s+(\d+);(?:\s*\/\*\s*([a-z]+(?:-\d+)?)\b)?/g;
  for (const match of root.matchAll(line)) {
    const [, name, r, g, b, palette] = match;
    const hex = [r, g, b]
      .map((channel) => Number(channel).toString(16).padStart(2, "0"))
      .join("");
    roles.push({ name, hex, palette: palette ?? null });
  }
  return roles;
}

let colorRoles: ColorRole[] | null = null;
function roles() {
  colorRoles ??= readColorRoles();
  return colorRoles;
}

// Which roles make sense for which utility. Builder roles are left out of
// these lists; they are named only when their exact colour matches.
const ROLE_FAMILIES: { utilities: RegExp; roles: RegExp }[] = [
  {
    utilities: /^(?:text|placeholder|caret|decoration|fill|stroke|accent)$/,
    roles:
      /^(?:foreground-.+|placeholder|action-foreground|brand|accent-strong|error|error-foreground|error-inverse|warning|warning-foreground|success|success-foreground|success-inverse|info|info-foreground|(?:orange|yellow|sky|pink)(?:-foreground)?|indigo)$/,
  },
  {
    utilities: /^(?:bg|from|via|to)$/,
    roles:
      /^(?:canvas|surface.*|overlay|scrim|action|action-hover|brand|brand-hover|accent-surface|error|error-hover|error-surface|warning|warning-surface|success|success-surface|info|info-surface|orange|yellow|sky|indigo|pink)$/,
  },
  {
    utilities: /^(?:border.*|divide.*|ring.*|outline)$/,
    roles:
      /^(?:separator|border-base|border-strong|focus|action|brand|error|warning-border|success|info|orange|yellow|sky|indigo|pink|scrim)$/,
  },
];

function familyRoles(utility: string) {
  const family = ROLE_FAMILIES.find((f) => f.utilities.test(utility));
  return roles().filter((role) =>
    family ? family.roles.test(role.name) : !BUILDER_ROLES.test(role.name),
  );
}

const REPLACEMENTS = buildReplacementMap();
const SIZES = buildSizeMap();

function exactToken(className: string) {
  const change = rewriteClassList(className, REPLACEMENTS, SIZES).changes[0];
  return change && !change.skipped ? change.to : null;
}

const COLOR_UTILITY =
  /^(?<variants>.*:)?(?<bang>!?)(?<utility>bg|text|border(?:-[xytrblse])?|ring(?:-offset)?|outline|divide(?:-[xy])?|fill|stroke|from|via|to|decoration|placeholder|caret|accent|shadow)-(?<value>.+?)(?<opacity>\/[\w.[\]%]+)?$/;

function hexOf(value: string) {
  const hex = /^\[#([0-9a-f]{3}|[0-9a-f]{6})\]$/i.exec(value)?.[1];
  if (hex)
    return (hex.length === 3 ? hex.replace(/./g, "$&$&") : hex).toLowerCase();
  const rgb =
    /^\[rgba?\((\d+)[,_ ]+(\d+)[,_ ]+(\d+)(?:[,_ /]+(?:1|1\.0|100%))?\)\]$/i.exec(
      value,
    );
  if (!rgb) return null;
  return rgb
    .slice(1, 4)
    .map((channel) => Number(channel).toString(16).padStart(2, "0"))
    .join("");
}

/** What to write instead of a raw or arbitrary colour, or null if not a colour. */
export function colorAdvice(className: string) {
  const parsed = COLOR_UTILITY.exec(className)?.groups;
  if (!parsed) return null;
  const { variants = "", bang, utility, value, opacity = "" } = parsed;
  const hex = value.startsWith("[") ? hexOf(value) : null;
  const palette = value.startsWith("[") ? null : value;
  if (!hex && !palette) return null;
  if (/(?:^|:)(?:dark|light):/.test(variants))
    return "It is also a dark:/light: class, which never renders: delete it.";
  const exact = exactToken(className);
  if (exact)
    return `Use "${exact}" instead: the design token with the same colour.`;
  const asClass = (role: string) =>
    `"${variants}${bang}${utility}-${role}${opacity}"`;
  const family = familyRoles(utility);
  const same = family.filter((role) =>
    hex ? role.hex === hex : role.palette === palette,
  );
  if (same.length > 0)
    return `Use ${same.map((role) => asClass(role.name)).join(" or ")} instead: the design token with the same colour.`;
  return `No ${utility}- token has this exact colour. Use the ${utility}- role that fits what the element means (${family.map((role) => role.name).join(", ")}) and check the screen, since the colour changes; or add a role in src/app/tokens/theme.css (and register it in globals.css's @theme).`;
}

// @shadcn/lint also reports any colour-like class that is not a theme token
// once the theme declares one (globals.css's `@theme`). With scanAllStrings it
// reads that into every string (`fill-mode-both`, `shadow-subtle`, SVG
// attributes in generated data), so only the palette and hard-coded colours
// are reported, as before the roles moved into `@theme`.
const PALETTE_CLASS_EXACT = new RegExp(
  colourClassPattern(PALETTE_VALUES).source,
);

function isPaletteOrHardcodedColor(report: Report, file: string) {
  if (String(report.messageId ?? "").startsWith("undeclaredToken"))
    return false;
  // no-raw-palette-classes and no-hex-colours report these, with the role to use
  // and ds-colour-exempt.json's exemptions.
  const said = [report.message, ...Object.values(report.data ?? {})].join(" ");
  if (PALETTE_CLASS_EXACT.test(said) || hexColours(said).length > 0)
    return false;
  const className = quotedClass(renderMessage(report, {}), report) ?? "";
  return !isWholeFileExempt(file) && !isExempt(file, className);
}

function quotedClass(message: string, report: Report) {
  const fromData = report.data?.className;
  if (typeof fromData === "string") return fromData;
  return /^"([^"]+)"/.exec(message)?.[1] ?? null;
}

// An arbitrary colour (`bg-[#fff]`) is no-arbitrary-colour's, which reads
// ds-colour-exempt.json; the shadcn rule reports the other arbitrary values.
function isNotArbitraryColour(report: Report) {
  const className = quotedClass(renderMessage(report, {}), report) ?? "";
  return arbitraryColours(className).length === 0;
}

const rewriteRawColor: Rewrite = (message, report) => {
  const className = quotedClass(message, report);
  const advice = className ? colorAdvice(className) : null;
  return advice && className
    ? `"${className}" uses the raw Tailwind palette. ${advice}`
    : message;
};

const rewriteArbitraryValue: Rewrite = (message, report) => {
  const className = quotedClass(message, report);
  if (!className) return message;
  const color = colorAdvice(className);
  if (color) return `"${className}" hardcodes a colour. ${color}`;
  const token = exactToken(className);
  if (token)
    return `"${className}" hardcodes a value. Use "${token}" instead: the design token with the same value.`;
  return message;
};

// ---------------------------------------------------------------------------
// Rules of our own.

function stringNodes(report: (node: Node, text: string) => void) {
  return {
    Literal(node: Node) {
      if (node.type === "Literal" && typeof node.value === "string")
        report(node, node.value);
    },
    TemplateElement(node: Node) {
      if (node.type === "TemplateElement")
        report(node, node.value.cooked ?? node.value.raw);
    },
  };
}

const THEME_VARIANT =
  /(?<![\w-])((?:[\w-]+:)*)(dark|light):(?=[\w[!*@(-])(\S*)/g;

const noDarkVariant: Rule.RuleModule = {
  meta: {
    type: "problem",
    docs: { description: "Disallow dark: and light: classes" },
    schema: [],
  },
  create(context) {
    return stringNodes((node, text) => {
      for (const match of text.matchAll(THEME_VARIANT)) {
        context.report({
          node,
          message: `"${match[0]}" is a ${match[2]}: class. The app has one theme: globals.css ties dark: to a .dark class that is never set, so this never renders. Delete it and keep the light value; the colour tokens (bg-surface, text-foreground-*, border-border-base) carry the theme.`,
        });
      }
    });
  },
};

// Tailwind's sizes (`text-sm`, `md:text-2xl`) and arbitrary ones
// (`text-[13px]`): the theme resets the first, so they render nothing, and the
// type scale (src/app/tokens/typography.css) replaces both (SECRT-3152).
const noRawTextSize: Rule.RuleModule = {
  meta: {
    type: "problem",
    docs: { description: "Disallow text sizes outside the type scale" },
    schema: [],
  },
  create(context) {
    return stringNodes((node, text) => {
      for (const match of text.matchAll(RAW_TEXT_SIZE)) {
        context.report({
          node,
          message: `"${match[0]}" is not a step of the type scale. ${rawTextSizeAdvice(match[2], match[3])}`,
        });
      }
    });
  },
};

const LEGACY_BUTTON_VARIANTS = new Map([
  ["legacy-default", 'variant="primary"'],
  ["legacy-destructive", 'variant="destructive"'],
  ["legacy-outline", 'variant="outline"'],
  ["legacy-ghost", 'variant="ghost"'],
  ["legacy-link", 'variant="link"'],
  ["legacy-sm", 'size="small"'],
  ["legacy-icon", 'size="icon"'],
]);

const noLegacyButtonVariant: Rule.RuleModule = {
  meta: {
    type: "problem",
    docs: { description: "Disallow the legacy Button variants" },
    schema: [],
  },
  create(context) {
    return {
      Literal(node: Node) {
        if (node.type !== "Literal" || typeof node.value !== "string") return;
        const replacement = LEGACY_BUTTON_VARIANTS.get(node.value);
        if (!replacement) return;
        context.report({
          node,
          message: `"${node.value}" renders the old LegacyButton. Use the design-system Button with ${replacement}, then check the screen: it does not look the same.`,
        });
      },
    };
  },
};

const BUTTON_INTERNALS = new Set([
  "extendedButtonVariants",
  "legacyButtonVariants",
  "compactButtonVariants",
]);

function legacyImportMessage(source: string) {
  const segments = source.replace(/\.[cm]?[jt]sx?$/, "").split("/");
  if (segments.includes("__legacy__") || segments.includes("legacy"))
    return `"${source}" is a legacy component. Use the design-system atom, molecule or organism in src/components instead (browse them in Storybook).`;
  const legacyModule = segments.find((segment) => /^Legacy[A-Z]/.test(segment));
  if (legacyModule === "LegacyButton")
    return `"${source}" is the old button. Render <Button> from @/components/atoms/Button/Button with a design-system variant and size.`;
  if (legacyModule)
    return `"${source}" is a legacy component. Use the design-system atom, molecule or organism in src/components instead.`;
  if (/^@\/components\/ui(?:\/|$)/.test(source))
    return `"${source}" held the raw shadcn primitives, which are gone. Import the atom, molecule or organism from src/components instead.`;
  return null;
}

function importSource(node: Node) {
  if (
    (node.type === "ImportDeclaration" ||
      node.type === "ExportNamedDeclaration" ||
      node.type === "ExportAllDeclaration" ||
      node.type === "ImportExpression") &&
    node.source?.type === "Literal" &&
    typeof node.source.value === "string"
  )
    return node.source.value;
  if (
    node.type === "CallExpression" &&
    node.callee.type === "Identifier" &&
    node.callee.name === "require" &&
    node.arguments[0]?.type === "Literal" &&
    typeof node.arguments[0].value === "string"
  )
    return node.arguments[0].value;
  return null;
}

function importedNames(node: Node) {
  if (node.type !== "ImportDeclaration") return [];
  return node.specifiers.flatMap((specifier) =>
    specifier.type === "ImportSpecifier"
      ? [
          specifier.imported.type === "Identifier"
            ? specifier.imported.name
            : String(specifier.imported.value),
        ]
      : [],
  );
}

function importVisitor(check: (node: Node, source: string) => void) {
  const visit = (node: Node) => {
    const source = importSource(node);
    if (source !== null) check(node, source);
  };
  return {
    ImportDeclaration: visit,
    ExportNamedDeclaration: visit,
    ExportAllDeclaration: visit,
    ImportExpression: visit,
    CallExpression: visit,
  };
}

const noLegacyImport: Rule.RuleModule = {
  meta: {
    type: "problem",
    docs: { description: "Disallow legacy component imports" },
    schema: [],
  },
  create(context) {
    return importVisitor((node, source) => {
      const message = legacyImportMessage(source);
      if (message) {
        context.report({ node, message });
        return;
      }
      if (!/(?:^|\/)atoms\/Button\/helpers$/.test(source)) return;
      for (const name of importedNames(node)) {
        if (!BUTTON_INTERNALS.has(name)) continue;
        context.report({
          node,
          message: `${name} is internal to the Button atom. Render <Button> with a variant and size instead.`,
        });
      }
    });
  },
};

const FOREIGN_ICONS =
  /^(?:lucide-react|@phosphor-icons\/react|react-icons|@radix-ui\/react-icons)(?:\/|$)/;
const ICON_ADVICE =
  "Import the icon from @hugeicons/core-free-icons and render it with the Icon atom: <Icon icon={Delete02Icon} size={16} /> (@/components/atoms/Icon/Icon), which applies the design-system stroke.";

const hugeiconsOnly: Rule.RuleModule = {
  meta: {
    type: "problem",
    docs: { description: "Require Hugeicons through the Icon atom" },
    schema: [],
  },
  create(context) {
    return importVisitor((node, source) => {
      if (FOREIGN_ICONS.test(source)) {
        context.report({
          node,
          message: `"${source}" is not the design-system icon set. ${ICON_ADVICE}`,
        });
        return;
      }
      if (
        source === "@hugeicons/react" &&
        importedNames(node).includes("HugeiconsIcon")
      )
        context.report({
          node,
          message: `Do not render HugeiconsIcon directly. ${ICON_ADVICE} Import only types such as IconSvgElement from @hugeicons/react.`,
        });
    });
  },
};

const DIRECTIVE = /^\s*(?:eslint|oxlint)-disable(?:-next-line|-line)?(?:\s|$)/;
const REASON = /\s--\s*\S/;

const requireSuppressionReason: Rule.RuleModule = {
  meta: {
    type: "problem",
    docs: { description: "Require a reason on lint suppressions" },
    schema: [],
  },
  create(context) {
    return {
      Program() {
        for (const comment of context.sourceCode.getAllComments()) {
          if (!DIRECTIVE.test(comment.value) || REASON.test(comment.value))
            continue;
          context.report({
            loc: comment.loc!,
            message: `Say why this rule is suppressed, on the same line after "--": // ${comment.value.trim()} -- <reason>.`,
          });
        }
      },
    };
  },
};

/**
 * Reports a file that has fewer violations than design-system-baseline.json
 * allows, so the entry can be lowered and the count cannot grow back.
 */
const baselineOutdated = {
  meta: {
    type: "suggestion",
    docs: { description: "Report baseline entries that can be lowered" },
    schema: [],
  },
  createOnce(context: Rule.RuleContext) {
    return {
      Program() {},
      after() {
        const file = relativeToFrontend(context.filename);
        const counts = countsByFile.get(file);
        countsByFile.delete(file);
        if (BASELINE_OFF || !counts) return;
        const lower = [...counts]
          .filter(([rule, count]) => count < allowedCount(file, rule))
          .map(
            ([rule, count]) =>
              `${rule} ${allowedCount(file, rule)} -> ${count}`,
          );
        if (lower.length === 0) return;
        context.report({
          loc: { line: 1, column: 0 },
          message: `Fewer design-system violations than the baseline allows (${lower.join(", ")}). Run \`pnpm lint:ds-baseline\` and commit scripts/eslint-rules/design-system-baseline.json so they cannot come back.`,
        });
      },
    };
  },
} as unknown as Rule.RuleModule;

// ---------------------------------------------------------------------------
// One colour system (SECRT-3154): the role tokens in src/app/tokens/theme.css.
// The shadcn names, the Tailwind palette and hex colours are failed in every
// string; ds-colour-exempt.json lists the artwork, charts and user-picked
// palettes that keep theirs.

const ALIAS_CLASS = colourClassPattern(
  Object.keys(ALIAS_ROLES)
    .sort((a, b) => b.length - a.length)
    .join("|"),
);

const noShadcnAliasColours: Rule.RuleModule = {
  meta: {
    type: "problem",
    docs: { description: "Disallow the shadcn colour names" },
    schema: [],
  },
  create(context) {
    return stringNodes((node, text) => {
      for (const match of text.matchAll(ALIAS_CLASS)) {
        const [whole, variants, utility, name, opacity = ""] = match;
        // `accent-` (accent-color) never takes a shadcn name in this app;
        // the word turns up in prose ("accent-foreground").
        if (utility === "accent") continue;
        const role = `${variants}${utility}-${ALIAS_ROLES[name]}${opacity}`;
        context.report({
          node,
          message: `"${whole}" is a shadcn colour name, which the app no longer has. Use "${role}", the role it aliased (same colour). The names left in src/app/tokens/theme.css are for streamdown's prebuilt markup only.`,
        });
      }
    });
  },
};

const PALETTE_CLASS = colourClassPattern(PALETTE_VALUES);
const PALETTE_TO_ROLE = buildPaletteMap();

const noRawPaletteClasses: Rule.RuleModule = {
  meta: {
    type: "problem",
    docs: { description: "Disallow the raw Tailwind palette" },
    schema: [],
  },
  create(context) {
    const file = relativeToFrontend(context.filename);
    if (isWholeFileExempt(file)) return {};
    return stringNodes((node, text) => {
      for (const match of text.matchAll(PALETTE_CLASS)) {
        const [whole, variants, utility, value, opacity = ""] = match;
        const bare = `${utility}-${value}`;
        if (isExempt(file, bare)) continue;
        const mapped = PALETTE_TO_ROLE.get(bare);
        const advice = mapped
          ? `Use "${variants}${mapped}${mapped.includes("/") ? "" : opacity}" (scripts/ds/palette-map.json).`
          : (colorAdvice(whole) ??
            "Use the colour role that fits what the element means.");
        context.report({
          node,
          message: `"${whole}" uses the raw Tailwind palette. ${advice} Artwork, charts and user-picked palettes go in ds-colour-exempt.json, with a reason.`,
        });
      }
    });
  },
};

// A hex colour: 3, 4, 6 or 8 digits, in any string (comments are not read).
const HEX_COLOUR =
  /(?<![\w&#/-])#(?:[0-9a-f]{8}|[0-9a-f]{6}|[0-9a-f]{3,4})(?![0-9a-z-])/gi;

export function hexColours(text: string) {
  return [...text.matchAll(HEX_COLOUR)].map((match) => match[0]);
}

const noHexColours: Rule.RuleModule = {
  meta: {
    type: "problem",
    docs: { description: "Disallow hex colours" },
    schema: [],
  },
  create(context) {
    const file = relativeToFrontend(context.filename);
    if (isWholeFileExempt(file)) return {};
    return stringNodes((node, text) => {
      // A hex that is a class's whole arbitrary value (`bg-[#fff]`,
      // `[color:#fff]`) is no-arbitrary-colour's, which names the role to
      // use; the same hex elsewhere in the string is still this rule's.
      const owned = new Set(
        arbitraryColours(text).map((match) => match.valueIndex),
      );
      for (const match of text.matchAll(HEX_COLOUR)) {
        const hex = match[0];
        if (isExempt(file, hex)) continue;
        if (owned.has(match.index)) continue;
        context.report({
          node,
          message: `"${hex}" hardcodes a colour. Use a role: a class (bg-surface, text-foreground-secondary) or, where a string needs a colour, rgb(var(--color-<role>)). Artwork, logos and chart series go in ds-colour-exempt.json, with a reason.`,
        });
      }
    });
  },
};

// A colour class whose arbitrary value is a colour literal: `bg-[#fff]`,
// `text-[rgb(20,20,20)]`, `border-[color:hsl(0_0%_90%)]/50`, or an arbitrary
// colour property (`[color:#fff]`). A value read from a role
// (`rgb(var(--color-surface))`) is not one, and neither is a colour inside a
// shadow or gradient (`shadow-[0_1px_2px_rgba(0,0,0,0.05)]`).
const COLOUR_LITERAL = String.raw`#[0-9a-f]{3,8}|(?:rgba?|hsla?|hwb|lab|lch|oklab|oklch|color)\((?![^\]\s]*var\()[^\]\s]*\)`;
const ARBITRARY_COLOUR_CLASS = new RegExp(
  String.raw`(?<![\w\-/\[.#])((?:[^\s"'\`{}]*:)?!?)(${[...COLOUR_UTILITIES].sort((a, b) => b.length - a.length).join("|")})-\[(?:color:)?(${COLOUR_LITERAL})\](\/(?:\d+(?:\.\d+)?|\[[^\]\s]+\]))?(?![\w\-])`,
  "gi",
);
const ARBITRARY_COLOUR_PROPERTY = new RegExp(
  String.raw`(?<![\w\-/\[.#])((?:[^\s"'\`{}]*:)?!?)\[((?:background-|border(?:-[a-z]+)?-|outline-|caret-|accent-|text-decoration-)?color|background|fill|stroke):(${COLOUR_LITERAL})\](?![\w\-])`,
  "gi",
);

/** Each arbitrary colour in `text`, with where its colour value starts. */
export function arbitraryColours(text: string) {
  return [
    ...text.matchAll(ARBITRARY_COLOUR_CLASS),
    ...text.matchAll(ARBITRARY_COLOUR_PROPERTY),
  ].map((match) => ({
    whole: match[0],
    value: match[3],
    // The value is the last bracketed part of the match.
    valueIndex: match.index + match[0].lastIndexOf(`${match[3]}]`),
  }));
}

const noArbitraryColour: Rule.RuleModule = {
  meta: {
    type: "problem",
    docs: { description: "Disallow arbitrary colour values" },
    schema: [],
  },
  create(context) {
    const file = relativeToFrontend(context.filename);
    if (isWholeFileExempt(file)) return {};
    return stringNodes((node, text) => {
      for (const { whole, value } of arbitraryColours(text)) {
        if (isExempt(file, value)) continue;
        const advice =
          colorAdvice(whole.replace("[color:", "[")) ??
          "Use the colour role that fits what the element means.";
        context.report({
          node,
          message: `"${whole}" hardcodes a colour. ${advice} Artwork, logos and chart series go in ds-colour-exempt.json, with a reason.`,
        });
      }
    });
  },
};

const shadcnRules = shadcn.rules as unknown as Record<string, Rule.RuleModule>;

const plugin = {
  meta: { name: PLUGIN_NAME },
  rules: {
    "no-raw-colors": ratchet(
      "no-raw-colors",
      shadcnRules["no-raw-colors"],
      rewriteRawColor,
      isPaletteOrHardcodedColor,
    ),
    "no-arbitrary-values": ratchet(
      "no-arbitrary-values",
      shadcnRules["no-arbitrary-values"],
      rewriteArbitraryValue,
      isNotArbitraryColour,
    ),
    "no-restyle": ratchet("no-restyle", shadcnRules["no-restyle"]),
    "no-inline-styles": ratchet(
      "no-inline-styles",
      shadcnRules["no-inline-styles"],
    ),
    "require-static-classes": ratchet(
      "require-static-classes",
      shadcnRules["require-static-classes"],
    ),
    "no-unknown-classes": ratchet(
      "no-unknown-classes",
      shadcnRules["no-unknown-classes"],
    ),
    "no-shadcn-alias-colours": ratchet(
      "no-shadcn-alias-colours",
      noShadcnAliasColours,
    ),
    "no-raw-palette-classes": ratchet(
      "no-raw-palette-classes",
      noRawPaletteClasses,
    ),
    "no-hex-colours": ratchet("no-hex-colours", noHexColours),
    "no-arbitrary-colour": ratchet("no-arbitrary-colour", noArbitraryColour),
    "no-dark-variant": ratchet("no-dark-variant", noDarkVariant),
    "no-raw-text-size": ratchet("no-raw-text-size", noRawTextSize),
    "no-legacy-import": ratchet("no-legacy-import", noLegacyImport),
    "no-legacy-button-variant": ratchet(
      "no-legacy-button-variant",
      noLegacyButtonVariant,
    ),
    "hugeicons-only": ratchet("hugeicons-only", hugeiconsOnly),
    "require-suppression-reason": ratchet(
      "require-suppression-reason",
      requireSuppressionReason,
    ),
    "baseline-outdated": baselineOutdated,
  },
};

export default plugin;
