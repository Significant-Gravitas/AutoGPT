import { readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import type { Rule } from "eslint";

// ESLint-compatible rules that oxlint loads as a JS plugin (.oxlintrc.json).
// Node strips the types when oxlint imports this file, so it stays one
// self-contained module: no relative imports, no TypeScript-only syntax.

type Node = Rule.Node;

const FRONTEND_ROOT = fileURLToPath(new URL("../..", import.meta.url));
const HOOK_NAME = /^use[A-Z0-9]/;
const MEMO_APIS = new Set(["useMemo", "useCallback", "memo"]);
const MEMO_DIRECTIVES = new Set(["use memo", "use no memo"]);
const MEMO_BOUNDARY_COMMENT = /^\s*memo-boundary:\s*\S/;
const FUNCTION_TYPES = new Set([
  "FunctionDeclaration",
  "FunctionExpression",
  "ArrowFunctionExpression",
]);

export const MANUAL_MEMO_BASELINE_FILE = fileURLToPath(
  new URL("./manual-memo-baseline.json", import.meta.url),
);

function readManualMemoBaseline(): Record<string, number> {
  return JSON.parse(readFileSync(MANUAL_MEMO_BASELINE_FILE, "utf8"));
}

function relativeToFrontend(filename: string) {
  return path.relative(FRONTEND_ROOT, filename).split(path.sep).join("/");
}

function calleeName(callee: Node) {
  if (callee.type === "Identifier") return callee.name;
  if (
    callee.type === "MemberExpression" &&
    !callee.computed &&
    callee.property.type === "Identifier"
  )
    return callee.property.name;
  return null;
}

function isHookName(name: string | null) {
  return name === "use" || (name !== null && HOOK_NAME.test(name));
}

function hasDirective(fn: Node, directive: string) {
  if (
    fn.type !== "FunctionDeclaration" &&
    fn.type !== "FunctionExpression" &&
    fn.type !== "ArrowFunctionExpression"
  )
    return false;
  if (fn.body.type !== "BlockStatement") return false;
  return fn.body.body.some(
    (statement) =>
      (statement as { directive?: string }).directive === directive,
  );
}

function functionName(node: Node) {
  if (node.type === "FunctionDeclaration") return node.id?.name ?? null;
  if (node.type === "FunctionExpression" && node.id) return node.id.name;
  const parent = node.parent;
  if (parent?.type === "VariableDeclarator" && parent.id.type === "Identifier")
    return parent.id.name;
  return null;
}

/**
 * A `use*` function that calls no hook is compiled by the React Compiler as
 * a plain function, so nothing it returns is memoised. Hook calls inside a
 * nested function belong to that function, not to the outer one. A function
 * that says "use no memo" is hook-named on purpose (its callers must not
 * cache the call), so it is left alone.
 */
const noHooklessUseFunction: Rule.RuleModule = {
  meta: {
    type: "problem",
    docs: { description: "Disallow `use*` functions that call no hook" },
    schema: [],
  },
  create(context) {
    const frames: { name: string | null; callsHook: boolean }[] = [];

    function enter(node: Node) {
      frames.push({
        name: functionName(node),
        callsHook: hasDirective(node, "use no memo"),
      });
    }

    function exit(node: Node) {
      const frame = frames.pop();
      if (!frame?.name || !HOOK_NAME.test(frame.name) || frame.callsHook)
        return;
      context.report({
        node,
        message: `\`${frame.name}\` calls no hook, so the React Compiler treats it as a plain function and memoises nothing it returns. Call a hook in it or drop the \`use\` prefix.`,
      });
    }

    return {
      FunctionDeclaration: enter,
      FunctionExpression: enter,
      ArrowFunctionExpression: enter,
      "FunctionDeclaration:exit": exit,
      "FunctionExpression:exit": exit,
      "ArrowFunctionExpression:exit": exit,
      CallExpression(node) {
        const frame = frames[frames.length - 1];
        if (frame && isHookName(calleeName(node.callee as Node)))
          frame.callsHook = true;
      },
    };
  },
};

function hasMemoBoundaryComment(
  context: Rule.RuleContext,
  node: Node,
): boolean {
  // From the call up to its statement, so a comment above an enclosing
  // function does not count for a memo() inside it.
  let current: Node | null = node;
  while (current && current.type !== "Program") {
    const comments = context.sourceCode.getCommentsBefore(current);
    if (comments.some((comment) => MEMO_BOUNDARY_COMMENT.test(comment.value)))
      return true;
    if (FUNCTION_TYPES.has(current.parent?.type ?? "")) return false;
    current = current.parent;
  }
  return false;
}

/**
 * The React Compiler memoises the folders it compiles, so hand-written
 * `useMemo`/`useCallback` there only adds noise, and the `"use memo"` /
 * `"use no memo"` directives opt a function in or out behind the compiler's
 * back. `memo()` stays for list-row boundaries, marked with a
 * `// memo-boundary: <why>` comment. Files written before the compiler keep
 * their existing count (manual-memo-baseline.json), which can only go down.
 */
const noManualMemo: Rule.RuleModule = {
  meta: {
    type: "problem",
    docs: { description: "Disallow manual memoisation in compiled folders" },
    schema: [],
  },
  create(context) {
    const reactLocals = new Map<string, string>();
    const reactNamespaces = new Set(["React"]);
    const calls: { node: Node; callee: Node }[] = [];

    function memoApi(callee: Node) {
      if (callee.type === "Identifier") return reactLocals.get(callee.name);
      if (
        callee.type === "MemberExpression" &&
        !callee.computed &&
        callee.object.type === "Identifier" &&
        reactNamespaces.has(callee.object.name) &&
        callee.property.type === "Identifier" &&
        MEMO_APIS.has(callee.property.name)
      )
        return callee.property.name;
      return undefined;
    }

    return {
      ImportDeclaration(node) {
        if (node.source.value !== "react") return;
        for (const specifier of node.specifiers) {
          if (specifier.type !== "ImportSpecifier") {
            reactNamespaces.add(specifier.local.name);
            continue;
          }
          const imported =
            specifier.imported.type === "Identifier"
              ? specifier.imported.name
              : String(specifier.imported.value);
          if (MEMO_APIS.has(imported))
            reactLocals.set(specifier.local.name, imported);
        }
      },
      ExpressionStatement(node) {
        const directive = (node as { directive?: string }).directive;
        if (!directive || !MEMO_DIRECTIVES.has(directive)) return;
        context.report({
          node,
          message: `"${directive}" overrides the React Compiler's own choice; fix what makes it bail out instead.`,
        });
      },
      CallExpression(node) {
        calls.push({ node, callee: node.callee as Node });
      },
      "Program:exit"(program) {
        const manual = calls.flatMap(({ node, callee }) => {
          const api = memoApi(callee);
          if (!api) return [];
          if (api === "memo" && hasMemoBoundaryComment(context, node))
            return [];
          return [{ node, api }];
        });
        const file = relativeToFrontend(context.filename);
        const allowed = readManualMemoBaseline()[file] ?? 0;
        if (manual.length < allowed) {
          context.report({
            node: program,
            message: `This file has ${manual.length} manual memo call(s) but manual-memo-baseline.json allows ${allowed}. Lower its entry to ${manual.length} (or delete it at 0) so the count cannot grow back.`,
          });
          return;
        }
        if (manual.length === allowed) return;
        for (const { node, api } of manual) {
          context.report({
            node,
            message:
              api === "memo"
                ? "The React Compiler memoises this folder. Keep `memo()` only at a list-row boundary, with a `// memo-boundary: <why>` comment above it."
                : `The React Compiler memoises this folder, so \`${api}\` is not needed here.${allowed > 0 ? ` This file is allowed ${allowed} existing manual memo call(s) and now has ${manual.length}.` : ""}`,
          });
        }
      },
    };
  },
};

const plugin = {
  meta: { name: "autogpt" },
  rules: {
    "no-hookless-use-function": noHooklessUseFunction,
    "no-manual-memo": noManualMemo,
  },
};

export default plugin;
