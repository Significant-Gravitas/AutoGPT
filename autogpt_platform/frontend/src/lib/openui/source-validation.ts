import { split, tokenize } from "@openuidev/lang-core";

const [
  leftParen,
  rightParen,
  leftBracket,
  rightBracket,
  leftBrace,
  rightBrace,
] = tokenize("()[]{}");
const delimiters = new Map([
  [leftParen.t, rightParen.t],
  [leftBracket.t, rightBracket.t],
  [leftBrace.t, rightBrace.t],
]);
const closing = new Set(delimiters.values());

export function validateSource(source: string) {
  const withoutComments = source.replace(
    /"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|#[^\n]*|\/\/[^\n]*/g,
    (token) => (token.startsWith('"') || token.startsWith("'") ? token : ""),
  );
  const tokens = tokenize(withoutComments);
  const expected = [];
  for (const token of tokens) {
    const close = delimiters.get(token.t);
    if (close !== undefined) {
      expected.push(close);
      if (expected.length > 64)
        throw new Error(
          "Workspace nesting exceeds 64 levels. Simplify the view.",
        );
    } else if (closing.has(token.t) && expected.pop() !== token.t) {
      throw new Error(
        "Mismatched brackets in source. Close each component and array correctly.",
      );
    }
  }
  if (expected.length)
    throw new Error(
      "Incomplete source. Close every component, array and object.",
    );
  const names = new Set<string>();
  for (const statement of split(tokens)) {
    if (names.has(statement.id))
      throw new Error(
        `Duplicate definition: ${statement.id}. Give each statement a unique name.`,
      );
    if (statement.id.startsWith("$"))
      throw new Error(
        "State bindings are not supported. Use literal values and chat follow-up actions.",
      );
    names.add(statement.id);
  }
}
