/**
 * Rewrites the Button atom's quiet ghost variants (`ghost-muted`,
 * `ghost-subtle`, `ghost-danger`, `ghost-warning`) to `variant="ghost"` with
 * a ghost tone, and adds the `shape` a size's corner radius used to come
 * from (xs, icon-xs, icon-2xs and icon-sm no longer set one), from the
 * tables in scripts/ds/button-variant-map.ts. Each rewrite renders the same
 * element and classes, so no pixel moves.
 *
 *   npx tsx scripts/ds/button-variants.ts <file|dir>...           # dry run
 *   npx tsx scripts/ds/button-variants.ts --write <file|dir>...   # rewrite
 *
 * It rewrites `<Button>` bound to the atom's import (not a parameter or local
 * of the same name) when `variant`, `size`, `shape` and `tone` are string
 * literals, as attributes or in a spread written as an object literal
 * (`{...{ variant: "ghost-muted" }}`), whose values are rewritten in place.
 * A conditional between literals that needs no change is left alone; one
 * that does, any value it can't read (a prop forwarded by a wrapper) and a
 * change to a prop written before a spread that can replace it are reported
 * for a hand migration and left as is. The new props go before any other
 * spread that precedes the prop they follow, so a spread still wins where it
 * did.
 */
import { readdirSync, readFileSync, statSync, writeFileSync } from "node:fs";
import { extname, join, relative } from "node:path";
import ts from "typescript";
import {
  migrateButtonStyle,
  type OldButtonStyle,
} from "./button-variant-map.ts";
import {
  importBindings,
  opaqueSpreadAfter,
  readProps,
  valueNode,
  writtenValue,
  type PropWriter,
} from "./jsx-codemod.ts";

const STYLE_PROPS = ["variant", "size", "shape", "tone"] as const;
type StyleProp = (typeof STYLE_PROPS)[number];

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

function importedButtons(file: ts.SourceFile) {
  const found = new Set<ts.ImportSpecifier>();
  for (const statement of file.statements) {
    if (
      !ts.isImportDeclaration(statement) ||
      !ts.isStringLiteral(statement.moduleSpecifier)
    )
      continue;
    const path = statement.moduleSpecifier.text;
    if (!/(?:^|\/)Button\/Button$/.test(path) && path !== "./Button") continue;
    const bindings = statement.importClause?.namedBindings;
    if (!bindings || !ts.isNamedImports(bindings)) continue;
    for (const element of bindings.elements) {
      if ((element.propertyName ?? element.name).text === "Button")
        found.add(element);
    }
  }
  return found;
}

// The values an expression can take, when it is a string literal, undefined
// or a conditional between them; null when it is anything else.
function possibleValues(
  expression: ts.Expression,
): (string | undefined)[] | null {
  if (
    ts.isStringLiteral(expression) ||
    ts.isNoSubstitutionTemplateLiteral(expression)
  )
    return [expression.text];
  if (ts.isIdentifier(expression) && expression.text === "undefined")
    return [undefined];
  if (ts.isParenthesizedExpression(expression))
    return possibleValues(expression.expression);
  if (ts.isConditionalExpression(expression)) {
    const whenTrue = possibleValues(expression.whenTrue);
    const whenFalse = possibleValues(expression.whenFalse);
    return whenTrue && whenFalse ? [...whenTrue, ...whenFalse] : null;
  }
  return null;
}

function writerValues(writer: PropWriter | undefined) {
  if (!writer) return [undefined];
  const value = writtenValue(writer);
  return value ? possibleValues(value) : null;
}

function staticValue(writer: PropWriter | undefined) {
  const values = writerValues(writer);
  return values?.length === 1 ? values[0] : null;
}

function combinations(values: Record<StyleProp, (string | undefined)[]>) {
  let result: OldButtonStyle[] = [{}];
  for (const prop of STYLE_PROPS)
    result = result.flatMap((style) =>
      values[prop].map((value) => ({ ...style, [prop]: value })),
    );
  return result;
}

function styleText(style: OldButtonStyle) {
  return STYLE_PROPS.flatMap((prop) =>
    style[prop] === undefined ? [] : [`${prop}="${style[prop]}"`],
  ).join(" ");
}

function rewriteElement(
  node: ts.JsxOpeningElement | ts.JsxSelfClosingElement,
  file: ts.SourceFile,
): { edits: Edit[]; change?: Change; manual?: Manual } {
  const line = file.getLineAndCharacterOfPosition(node.getStart(file)).line + 1;
  const props = readProps(node.attributes);
  const { writers } = props;
  const writerText = (prop: StyleProp) =>
    writers.get(prop)?.node.getText(file) ?? prop;
  const values = {} as Record<StyleProp, (string | undefined)[]>;
  const unreadable: StyleProp[] = [];
  for (const prop of STYLE_PROPS) {
    const read = writerValues(writers.get(prop));
    if (read) values[prop] = read;
    else {
      unreadable.push(prop);
      values[prop] = [undefined];
    }
  }
  // An unreadable shape or tone only matters next to a value that needs one.
  const shapeSet = staticValue(writers.get("shape")) != null;
  const blocking = unreadable.filter(
    (prop) => prop === "variant" || (prop === "size" && !shapeSet),
  );
  if (blocking.length > 0)
    return {
      edits: [],
      manual: {
        line,
        reason: `can't read ${blocking.map(writerText).join(", ")}`,
      },
    };

  const styles = combinations(values);
  const migrations = styles.map((style) => ({
    style,
    migration: migrateButtonStyle(style),
  }));
  const needed = migrations.filter(
    ({ migration }) =>
      !("props" in migration) || Object.keys(migration.props).length > 0,
  );
  if (needed.length === 0) return { edits: [] };
  // A spread whose keys can't be read, after a style prop, can replace it,
  // so the change it needs is not known.
  const replaceable = STYLE_PROPS.flatMap((prop) => {
    const spread = opaqueSpreadAfter(props, writers.get(prop));
    return spread ? [`${writerText(prop)} before ${spread.getText(file)}`] : [];
  });
  if (replaceable.length > 0)
    return {
      edits: [],
      manual: { line, reason: `can't read ${replaceable.join(", ")}` },
    };
  const dynamic = STYLE_PROPS.filter(
    (prop) => values[prop].length > 1 || unreadable.includes(prop),
  );
  // A dynamic value is fine when every value it can take needs the same
  // change, to a prop that is not itself dynamic.
  const first = migrations[0].migration;
  const uniform =
    "props" in first &&
    !dynamic.some((prop) => prop in first.props) &&
    migrations.every(
      ({ migration }) => JSON.stringify(migration) === JSON.stringify(first),
    );
  if (dynamic.length > 0 && !uniform)
    return {
      edits: [],
      manual: {
        line,
        reason: `dynamic ${dynamic.join(", ")}: ${needed.map(({ style }) => styleText(style)).join(" | ")} need a change`,
      },
    };
  const [{ style, migration }] = migrations;
  if (!("props" in migration))
    return { edits: [], manual: { line, reason: migration.conflict } };

  const edits: Edit[] = [];
  // Inserted text by position; `beforeSpread` text goes in front of a spread.
  const added = new Map<number, { beforeSpread: boolean; texts: string[] }>();
  const firstOpaque = props.opaque[0];
  function add(writer: PropWriter | undefined, text: string) {
    const after = writer?.owner;
    const beforeSpread =
      !!firstOpaque &&
      (!after || firstOpaque.getStart(file) < after.getStart(file));
    const position = beforeSpread
      ? firstOpaque.getStart(file)
      : (after?.getEnd() ?? node.tagName.getEnd());
    const entry = added.get(position) ?? { beforeSpread, texts: [] };
    added.set(position, { ...entry, texts: [...entry.texts, text] });
  }
  for (const prop of STYLE_PROPS) {
    const value = migration.props[prop];
    if (value === undefined) continue;
    const writer = writers.get(prop);
    const current = writer && valueNode(writer);
    if (current) {
      edits.push({
        start: current.getStart(file),
        end: current.getEnd(),
        text: `"${value}"`,
      });
      continue;
    }
    const anchor =
      prop === "shape"
        ? (writers.get("size") ?? writers.get("variant"))
        : writers.get("variant");
    add(anchor, `${prop}="${value}"`);
  }
  for (const [position, { beforeSpread, texts }] of added)
    edits.push({
      start: position,
      end: position,
      text: beforeSpread ? `${texts.join(" ")} ` : ` ${texts.join(" ")}`,
    });
  return {
    edits,
    change: {
      line,
      from:
        dynamic.length > 0
          ? `${styleText(style)} (and the other values of ${dynamic.join(", ")})`
          : styleText(style),
      to: styleText({ ...style, ...migration.props }),
    },
  };
}

export function rewriteButtonVariants(source: string, fileName: string) {
  const file = ts.createSourceFile(
    fileName,
    source,
    ts.ScriptTarget.Latest,
    true,
    fileName.endsWith(".tsx") ? ts.ScriptKind.TSX : ts.ScriptKind.TS,
  );
  const buttons = importedButtons(file);
  const bindingOf = buttons.size > 0 ? importBindings(file) : () => undefined;
  const edits: Edit[] = [];
  const changes: Change[] = [];
  const manual: Manual[] = [];
  function isButton(tag: ts.Identifier) {
    const binding = bindingOf(tag);
    return binding !== undefined && buttons.has(binding);
  }
  function visit(node: ts.Node) {
    if (
      (ts.isJsxOpeningElement(node) || ts.isJsxSelfClosingElement(node)) &&
      ts.isIdentifier(node.tagName) &&
      isButton(node.tagName)
    ) {
      const result = rewriteElement(node, file);
      edits.push(...result.edits);
      if (result.change) changes.push(result.change);
      if (result.manual) manual.push(result.manual);
    }
    ts.forEachChild(node, visit);
  }
  if (buttons.size > 0) visit(file);
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
      "usage: npx tsx scripts/ds/button-variants.ts [--write] <file|dir>...",
    );
    return 2;
  }
  let total = 0;
  let touched = 0;
  const manual: string[] = [];
  for (const file of targets.flatMap((target) => [...sourceFiles(target)])) {
    const source = readFileSync(file, "utf8");
    const result = rewriteButtonVariants(source, file);
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
    console.log(`\n${manual.length} site(s) to check by hand:`);
    for (const line of manual) console.log(`  ${line}`);
  }
  console.log(
    `${total} rewrite(s) in ${touched} file(s)${write ? "" : " (dry run, pass --write to apply)"}`,
  );
  return manual.length > 0 ? 1 : 0;
}

if (process.argv[1]?.endsWith("button-variants.ts")) {
  process.exitCode = main(process.argv.slice(2));
}
