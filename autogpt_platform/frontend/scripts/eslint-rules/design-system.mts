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
  keep: (report: Report) => boolean = () => true,
): Rule.RuleModule {
  const messages = (inner.meta?.messages ?? {}) as Record<string, string>;
  return {
    meta: { ...inner.meta },
    create(context) {
      const reports: Report[] = [];
      const target = context as unknown as Record<PropertyKey, unknown>;
      const collecting = new Proxy(
        {},
        {
          get(_, key) {
            if (key === "report")
              return (report: Report) => {
                if (keep(report)) reports.push(report);
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
          const file = relativeToFrontend(context.filename);
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
// Colour roles, read from src/app/globals.css so the advice never drifts.

const SHADCN_NAMES =
  /^(?:background|foreground|card|card-foreground|popover|popover-foreground|primary|primary-foreground|secondary|secondary-foreground|muted|muted-foreground|accent|accent-foreground|destructive|destructive-foreground|border|input|ring|sidebar.*)$/;
const BUILDER_ROLES =
  /^(?:builder-canvas|builder-grid|node-surface|node-border|node-selected|edge|handle)$/;

interface ColorRole {
  name: string;
  hex: string;
  palette: string | null;
}

function readColorRoles(): ColorRole[] {
  const css = readFileSync(
    path.join(FRONTEND_ROOT, "src/app/globals.css"),
    "utf8",
  );
  const start = css.indexOf(":root {");
  const root = css.slice(start, css.indexOf("\n}", start));
  const roles: ColorRole[] = [];
  const line =
    /--color-([a-z-]+):\s*(\d+)\s+(\d+)\s+(\d+);(?:\s*\/\*\s*([a-z]+(?:-\d+)?)\b)?/g;
  for (const match of root.matchAll(line)) {
    const [, name, r, g, b, palette] = match;
    if (SHADCN_NAMES.test(name)) continue;
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
      /^(?:foreground-.+|placeholder|action-foreground|brand|accent-strong|error|error-foreground|warning|warning-foreground|success|success-foreground|info|info-foreground)$/,
  },
  {
    utilities: /^(?:bg|from|via|to)$/,
    roles:
      /^(?:canvas|surface.*|overlay|action|action-hover|brand|brand-hover|accent-surface|error|error-hover|error-surface|warning|warning-surface|success|success-surface|info|info-surface)$/,
  },
  {
    utilities: /^(?:border.*|divide.*|ring.*|outline)$/,
    roles:
      /^(?:separator|border-base|border-strong|focus|action|brand|error|warning-border|success|info)$/,
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
  return `No ${utility}- token has this exact colour. Use the ${utility}- role that fits what the element means (${family.map((role) => role.name).join(", ")}) and check the screen, since the colour changes; or add a role in src/app/globals.css (its :root and @theme).`;
}

// @shadcn/lint also reports any colour-like class that is not a theme token
// once the theme declares one (globals.css's `@theme`). With scanAllStrings it
// reads that into every string (`fill-mode-both`, `shadow-subtle`, SVG
// attributes in generated data), so only the palette and hard-coded colours
// are reported, as before the roles moved into `@theme`.
function isPaletteOrHardcodedColor(report: Report) {
  return !String(report.messageId ?? "").startsWith("undeclaredToken");
}

function quotedClass(message: string, report: Report) {
  const fromData = report.data?.className;
  if (typeof fromData === "string") return fromData;
  return /^"([^"]+)"/.exec(message)?.[1] ?? null;
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
    "no-dark-variant": ratchet("no-dark-variant", noDarkVariant),
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
