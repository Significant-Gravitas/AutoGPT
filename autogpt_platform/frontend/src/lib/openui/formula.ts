type Expression =
  | { kind: "number"; value: number }
  | { kind: "field"; name: string }
  | { kind: "count"; name: string }
  | {
      kind: "operation";
      operator: string;
      left: Expression;
      right: Expression;
    };

export function parseFormula(formula: string) {
  if (formula.length > 500)
    throw new Error("Use a formula of at most 500 characters.");
  const tokens: string[] = [];
  const pattern =
    /\s*((?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?|[a-z_][a-z_0-9]*|[()+\-*/])/giy;
  let end = 0;
  let match: RegExpExecArray | null;
  while ((match = pattern.exec(formula))) {
    tokens.push(match[1]);
    end = pattern.lastIndex;
  }
  if (formula.slice(end).trim() || tokens.length > 128)
    throw new Error(
      "Use only field names, numbers, + - * /, parentheses, and count(field).",
    );
  let index = 0;
  const references: { name: string; kind: "field" | "count" }[] = [];
  function expression(minimum = 0, depth = 0): Expression {
    if (depth > 24) throw new Error("Simplify the nested formula.");
    const token = tokens[index++];
    let left: Expression;
    if (token === "(") {
      left = expression(0, depth + 1);
      if (tokens[index++] !== ")")
        throw new Error("Close the formula's parentheses.");
    } else if (token === "-" || token === "+") {
      left = {
        kind: "operation",
        operator: token,
        left: { kind: "number", value: 0 },
        right: expression(3, depth + 1),
      };
    } else if (
      token &&
      /^[\d.]/.test(token) &&
      Number.isFinite(Number(token))
    ) {
      left = { kind: "number", value: Number(token) };
    } else if (token && /^[a-z_][a-z_0-9]*$/i.test(token)) {
      const counting = token === "count" && tokens[index] === "(";
      if (counting) index++;
      const name = counting ? tokens[index++] : token;
      if (
        !name ||
        !/^[a-z_][a-z_0-9]*$/i.test(name) ||
        (counting && tokens[index++] !== ")")
      )
        throw new Error(
          "Use count(field_name) for a multiple selection field.",
        );
      left = { kind: counting ? "count" : "field", name };
      references.push({ name, kind: counting ? "count" : "field" });
    } else throw new Error("The formula needs a number or a field name.");
    while (index < tokens.length) {
      const operator = tokens[index];
      const precedence =
        operator === "+" || operator === "-"
          ? 1
          : operator === "*" || operator === "/"
            ? 2
            : 0;
      if (!precedence || precedence < minimum) break;
      index++;
      left = {
        kind: "operation",
        operator,
        left,
        right: expression(precedence + 1, depth + 1),
      };
    }
    return left;
  }
  const root = expression();
  if (index !== tokens.length)
    throw new Error("Check the formula's operators and parentheses.");
  return { root, references };
}

export function evaluateFormula(
  node: Expression,
  get: (name: string) => unknown,
): number | null {
  if (node.kind === "number") return node.value;
  if (node.kind === "field" || node.kind === "count") {
    const value = get(node.name);
    if (node.kind === "count")
      return Array.isArray(value) ? value.length : null;
    if (
      typeof value !== "number" &&
      (typeof value !== "string" ||
        !/^[+-]?(?:\d+\.?\d*|\.\d+)(?:e[+-]?\d+)?$/i.test(value.trim()))
    )
      return null;
    return Number.isFinite(Number(value)) ? Number(value) : null;
  }
  const left = evaluateFormula(node.left, get);
  const right = evaluateFormula(node.right, get);
  if (left === null || right === null) return null;
  if (node.operator === "/" && right === 0)
    throw new Error("Cannot divide by zero.");
  const value =
    node.operator === "+"
      ? left + right
      : node.operator === "-"
        ? left - right
        : node.operator === "*"
          ? left * right
          : left / right;
  if (!Number.isFinite(value))
    throw new Error("The result is too large to calculate.");
  return value;
}
