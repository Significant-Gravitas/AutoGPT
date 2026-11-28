/**
 * Rewrites the Text atom's old variant names (`small-medium-tight`,
 * `body-mono-tight`, ...) to a role plus the props that carry what the name
 * baked in, from the table in scripts/ds/text-variant-map.ts. Each rewrite
 * renders the same classes and element, so no pixel moves.
 *
 *   npx tsx scripts/ds/text-variants.ts <file|dir>...           # dry run
 *   npx tsx scripts/ds/text-variants.ts --write <file|dir>...   # rewrite
 *
 * It rewrites `variant` and `size` on `<Text>` and `<OverflowText>`, bound to
 * the atom's import (not a parameter or local of the same name), when they
 * are string literals. A spread written as an object literal
 * (`{...{ variant: "body-medium" }}`) is read like the attributes, and the
 * values in it are rewritten in place. A prop already at the call site wins
 * over the old variant's, as it did before, so it is kept and the baked one
 * is not added. The new props go before any other spread that precedes
 * `variant`, so a spread still wins where it did. A dynamic `variant`/`size`,
 * one written before a spread that can replace it, or a dynamic prop the old
 * variant baked in, is reported for a hand migration and left as is.
 */
import { readdirSync, readFileSync, statSync, writeFileSync } from "node:fs";
import { extname, join, relative } from "node:path";
import ts from "typescript";
import {
  isOldTextVariant,
  TEXT_VARIANT_MAP,
  type TextVariantMapping,
} from "./text-variant-map.ts";
import {
  importBindings,
  opaqueSpreadAfter,
  readProps,
  valueNode,
  writtenValue,
  type PropWriter,
} from "./jsx-codemod.ts";

interface Component {
  /** Whether `as` decides its element; OverflowText always renders a span. */
  takesAs: boolean;
  /** The variant it renders without one. */
  defaultVariant?: string;
}

const COMPONENTS: Record<string, Component> = {
  "Text/Text": { takesAs: true },
  "OverflowText/OverflowText": {
    takesAs: false,
    defaultVariant: "body",
  },
};

export interface Change {
  line: number;
  from: string;
  to: string;
}

export interface Manual {
  line: number;
  reason: string;
}

interface Edit {
  start: number;
  end: number;
  text: string;
}

function importedComponents(file: ts.SourceFile) {
  const found = new Map<ts.ImportSpecifier, Component>();
  for (const statement of file.statements) {
    if (
      !ts.isImportDeclaration(statement) ||
      !ts.isStringLiteral(statement.moduleSpecifier)
    )
      continue;
    const path = statement.moduleSpecifier.text;
    const entry = Object.entries(COMPONENTS).find(([suffix]) =>
      `/${path}`.endsWith(`/${suffix}`),
    );
    const bindings = statement.importClause?.namedBindings;
    if (!entry || !bindings || !ts.isNamedImports(bindings)) continue;
    const exported = entry[0].split("/").pop();
    for (const element of bindings.elements) {
      if ((element.propertyName ?? element.name).text === exported)
        found.set(element, entry[1]);
    }
  }
  return found;
}

function staticString(writer: PropWriter) {
  const value = writtenValue(writer);
  return value &&
    (ts.isStringLiteral(value) || ts.isNoSubstitutionTemplateLiteral(value))
    ? value.text
    : undefined;
}

// A value written at the call site that the reader can see: a string, a
// boolean shorthand or a literal `{true}` / `{false}`.
function isStaticValue(writer: PropWriter) {
  const value = writtenValue(writer);
  return (
    !value ||
    staticString(writer) !== undefined ||
    value.kind === ts.SyntaxKind.TrueKeyword ||
    value.kind === ts.SyntaxKind.FalseKeyword
  );
}

function attributeText(name: string, value: string | boolean) {
  return value === true ? name : `${name}="${value}"`;
}

function rewriteElement(
  node: ts.JsxOpeningElement | ts.JsxSelfClosingElement,
  component: Component,
  file: ts.SourceFile,
): { edits: Edit[]; change?: Change; manual?: Manual } {
  const line = file.getLineAndCharacterOfPosition(node.getStart(file)).line + 1;
  const props = readProps(node.attributes);
  const { writers } = props;
  const variantWriter = writers.get("variant");
  const sizeWriter = writers.get("size");
  const variant = variantWriter
    ? staticString(variantWriter)
    : component.defaultVariant;
  const size = sizeWriter ? staticString(sizeWriter) : undefined;
  if (
    (variantWriter && variant === undefined) ||
    (sizeWriter && size === undefined)
  )
    return {
      edits: [],
      manual: {
        line,
        reason: `dynamic ${variantWriter && variant === undefined ? "variant" : "size"}: ${(variant === undefined ? variantWriter : sizeWriter)?.node.getText(file)}`,
      },
    };
  if (!variant || !isOldTextVariant(variant)) return { edits: [] };
  if (size !== undefined && !isOldTextVariant(size)) return { edits: [] };

  const elementMapping: TextVariantMapping = TEXT_VARIANT_MAP[variant];
  const classMapping: TextVariantMapping = TEXT_VARIANT_MAP[size ?? variant];
  const added: [string, string | boolean][] = [];
  if (component.takesAs && elementMapping.as && !writers.has("as"))
    added.push(["as", elementMapping.as]);
  for (const [prop, value] of Object.entries(classMapping.props ?? {})) {
    const existing = writers.get(prop);
    if (!existing) {
      added.push([prop, value]);
      continue;
    }
    if (!isStaticValue(existing))
      return {
        edits: [],
        manual: {
          line,
          reason: `dynamic ${existing.node.getText(file)} next to ${size ?? variant}'s ${attributeText(prop, value)}`,
        },
      };
  }

  const renamedVariant = elementMapping.variant !== variant;
  const renamedSize = size !== undefined && classMapping.variant !== size;
  if (!renamedVariant && !renamedSize && added.length === 0)
    return { edits: [] };
  // A spread after the variant or size whose keys can't be read can replace
  // it, so the change it needs is not known.
  for (const writer of [variantWriter, sizeWriter]) {
    const spread = opaqueSpreadAfter(props, writer);
    if (spread)
      return {
        edits: [],
        manual: {
          line,
          reason: `can't read ${writer?.node.getText(file)} before ${spread.getText(file)}`,
        },
      };
  }
  const edits: Edit[] = [];
  const variantValue = variantWriter && valueNode(variantWriter);
  if (variantValue && renamedVariant)
    edits.push({
      start: variantValue.getStart(file),
      end: variantValue.getEnd(),
      text: `"${elementMapping.variant}"`,
    });
  const sizeValue = sizeWriter && valueNode(sizeWriter);
  if (sizeValue && renamedSize)
    edits.push({
      start: sizeValue.getStart(file),
      end: sizeValue.getEnd(),
      text: `"${classMapping.variant}"`,
    });
  if (added.length > 0) {
    const anchor = (sizeWriter ?? variantWriter)?.owner;
    const firstOpaque = props.opaque[0];
    const text = added
      .map(([prop, value]) => attributeText(prop, value))
      .join(" ");
    if (
      anchor &&
      !(firstOpaque && firstOpaque.getStart(file) < anchor.getStart(file))
    )
      edits.push({
        start: anchor.getEnd(),
        end: anchor.getEnd(),
        text: ` ${text}`,
      });
    else if (firstOpaque)
      edits.push({
        start: firstOpaque.getStart(file),
        end: firstOpaque.getStart(file),
        text: `${text} `,
      });
    else
      edits.push({
        start: node.tagName.getEnd(),
        end: node.tagName.getEnd(),
        text: ` ${text}`,
      });
  }
  if (edits.length === 0) return { edits };

  const from = [
    variantWriter ? `variant="${variant}"` : "",
    size !== undefined ? `size="${size}"` : "",
  ]
    .filter(Boolean)
    .join(" ");
  const to = [
    variantWriter ? `variant="${elementMapping.variant}"` : "",
    size !== undefined ? `size="${classMapping.variant}"` : "",
    ...added.map(([prop, value]) => attributeText(prop, value)),
  ]
    .filter(Boolean)
    .join(" ");
  return { edits, change: { line, from, to } };
}

export function rewriteTextVariants(source: string, fileName: string) {
  const file = ts.createSourceFile(
    fileName,
    source,
    ts.ScriptTarget.Latest,
    true,
    fileName.endsWith(".tsx") ? ts.ScriptKind.TSX : ts.ScriptKind.TS,
  );
  const components = importedComponents(file);
  const bindingOf =
    components.size > 0 ? importBindings(file) : () => undefined;
  const edits: Edit[] = [];
  const changes: Change[] = [];
  const manual: Manual[] = [];
  function visit(node: ts.Node) {
    if (
      (ts.isJsxOpeningElement(node) || ts.isJsxSelfClosingElement(node)) &&
      ts.isIdentifier(node.tagName)
    ) {
      const binding = bindingOf(node.tagName);
      const component = binding && components.get(binding);
      if (component) {
        const result = rewriteElement(node, component, file);
        edits.push(...result.edits);
        if (result.change) changes.push(result.change);
        if (result.manual) manual.push(result.manual);
      }
    }
    ts.forEachChild(node, visit);
  }
  if (components.size > 0) visit(file);
  let output = source;
  for (const edit of edits.sort((a, b) => b.start - a.start))
    output = output.slice(0, edit.start) + edit.text + output.slice(edit.end);
  return { output, changes, manual };
}

const SOURCE_EXTENSIONS = new Set([".tsx"]);
const SKIP_DIRS = new Set(["node_modules", ".next", "__generated__"]);

function* sourceFiles(path: string): Generator<string> {
  if (statSync(path).isFile()) {
    if (SOURCE_EXTENSIONS.has(extname(path))) yield path;
    return;
  }
  for (const entry of readdirSync(path)) {
    if (SKIP_DIRS.has(entry)) continue;
    yield* sourceFiles(join(path, entry));
  }
}

function main(argv: string[]) {
  const write = argv.includes("--write");
  const targets = argv.filter((arg) => !arg.startsWith("--"));
  if (targets.length === 0) {
    console.error(
      "usage: npx tsx scripts/ds/text-variants.ts [--write] <file|dir>...",
    );
    return 2;
  }
  let total = 0;
  let touched = 0;
  const manual: string[] = [];
  for (const file of targets.flatMap((target) => [...sourceFiles(target)])) {
    const source = readFileSync(file, "utf8");
    const result = rewriteTextVariants(source, file);
    const path = relative(process.cwd(), file);
    for (const item of result.manual)
      manual.push(`${path}:${item.line}  ${item.reason}`);
    if (result.changes.length === 0) continue;
    touched++;
    for (const change of result.changes) {
      total++;
      console.log(`${path}:${change.line}  ${change.from} -> ${change.to}`);
    }
    if (write) writeFileSync(file, result.output);
  }
  if (manual.length > 0) {
    console.log(`\n${manual.length} site(s) to migrate by hand:`);
    for (const line of manual) console.log(`  ${line}`);
  }
  console.log(
    `${total} rewrite(s) in ${touched} file(s)${write ? "" : " (dry run, pass --write to apply)"}`,
  );
  return manual.length > 0 ? 1 : 0;
}

if (process.argv[1]?.endsWith("text-variants.ts")) {
  process.exitCode = main(process.argv.slice(2));
}
