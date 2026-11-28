/**
 * What the JSX codemods (scripts/ds/text-variants.ts and
 * scripts/ds/button-variants.ts) share: which import a tag is bound to, who
 * sets each prop, and which spread can replace a prop written before it.
 */
import ts from "typescript";

/**
 * Resolves a JSX tag to the import specifier it is bound to, or undefined
 * when it names something else: a parameter, a local or a component of the
 * same name declared in the file.
 */
export function importBindings(file: ts.SourceFile) {
  const options: ts.CompilerOptions = {
    noLib: true,
    noResolve: true,
    types: [],
    jsx: ts.JsxEmit.Preserve,
  };
  const host = ts.createCompilerHost(options);
  const program = ts.createProgram({
    rootNames: [file.fileName],
    options,
    host: {
      ...host,
      getSourceFile: (name) => (name === file.fileName ? file : undefined),
      fileExists: (name) => name === file.fileName,
      readFile: () => undefined,
    },
  });
  const checker = program.getTypeChecker();
  return (tag: ts.Identifier) =>
    checker.getSymbolAtLocation(tag)?.declarations?.find(ts.isImportSpecifier);
}

/**
 * What sets a prop: an attribute, or a property of a spread written as an
 * object literal (`{...{ size: "xs" }}`). `owner` is the attribute or spread
 * itself.
 */
export type PropWriter =
  | { node: ts.JsxAttribute; owner: ts.JsxAttribute }
  | {
      node: ts.PropertyAssignment | ts.ShorthandPropertyAssignment;
      owner: ts.JsxSpreadAttribute;
    };

interface ElementProps {
  /** The writer that wins for each prop: the last one written. */
  writers: Map<string, PropWriter>;
  /** The spreads whose keys can't be read (`{...props}`): any prop. */
  opaque: ts.JsxSpreadAttribute[];
}

function propertyName(name: ts.PropertyName) {
  return ts.isIdentifier(name) ||
    ts.isStringLiteral(name) ||
    ts.isNoSubstitutionTemplateLiteral(name)
    ? name.text
    : undefined;
}

// The keys and properties of a spread written as an object literal, or
// undefined when any key can't be read (a nested spread, a computed key).
function spreadProperties(spread: ts.JsxSpreadAttribute) {
  let expression = spread.expression;
  while (
    ts.isParenthesizedExpression(expression) ||
    ts.isAsExpression(expression) ||
    ts.isSatisfiesExpression(expression)
  )
    expression = expression.expression;
  if (!ts.isObjectLiteralExpression(expression)) return undefined;
  const found: [
    string,
    ts.PropertyAssignment | ts.ShorthandPropertyAssignment,
  ][] = [];
  for (const property of expression.properties) {
    if (
      !ts.isPropertyAssignment(property) &&
      !ts.isShorthandPropertyAssignment(property)
    )
      return undefined;
    const name = propertyName(property.name);
    if (name === undefined) return undefined;
    found.push([name, property]);
  }
  return found;
}

/** Who sets each prop, reading the spreads that are object literals. */
export function readProps(attributes: ts.JsxAttributes): ElementProps {
  const writers = new Map<string, PropWriter>();
  const opaque: ts.JsxSpreadAttribute[] = [];
  for (const property of attributes.properties) {
    if (ts.isJsxAttribute(property)) {
      if (ts.isIdentifier(property.name))
        writers.set(property.name.text, { node: property, owner: property });
      continue;
    }
    const properties = spreadProperties(property);
    if (!properties) {
      opaque.push(property);
      continue;
    }
    for (const [name, node] of properties)
      writers.set(name, { node, owner: property });
  }
  return { writers, opaque };
}

/**
 * The expression a writer gives its prop; null for a bare attribute
 * (`mono`), which passes true.
 */
export function writtenValue(writer: PropWriter): ts.Expression | null {
  const { node } = writer;
  if (ts.isShorthandPropertyAssignment(node)) return node.name;
  if (ts.isPropertyAssignment(node)) return node.initializer;
  if (!node.initializer) return null;
  return ts.isJsxExpression(node.initializer)
    ? (node.initializer.expression ?? node.initializer)
    : node.initializer;
}

/** The node holding a writer's value, to replace; undefined when it has none. */
export function valueNode(writer: PropWriter) {
  const { node } = writer;
  return ts.isShorthandPropertyAssignment(node) ? undefined : node.initializer;
}

/**
 * The first spread whose keys can't be read written after `writer`, which
 * can replace its value.
 */
export function opaqueSpreadAfter(
  props: ElementProps,
  writer: PropWriter | undefined,
) {
  if (!writer) return undefined;
  return props.opaque.find((spread) => spread.pos > writer.owner.pos);
}
