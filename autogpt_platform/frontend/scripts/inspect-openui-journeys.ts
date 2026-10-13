import { readFileSync, readdirSync, writeFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { parseWorkspace, workspaceText } from "../src/lib/openui/parse";
import type { ElementNode } from "@openuidev/lang-core";

function nodes(value: unknown): ElementNode[] {
  if (Array.isArray(value)) return value.flatMap(nodes);
  if (!value || typeof value !== "object") return [];
  const children = Object.values(value).flatMap(nodes);
  return "typeName" in value && "props" in value
    ? [value as ElementNode, ...children]
    : children;
}
function files(path: string): string[] {
  return readdirSync(path, { withFileTypes: true }).flatMap((entry) => {
    const full = join(path, entry.name);
    return entry.isDirectory()
      ? files(full)
      : /^turn-\d+\.json$/.test(entry.name)
        ? [full]
        : [];
  });
}
const root = resolve(process.argv[2]);
const results = files(root)
  .sort()
  .map((path) => {
    const result = JSON.parse(readFileSync(path, "utf8"));
    const views = (result.views as { source: string; summary: string }[]).map(
      (view) => {
        try {
          const root = parseWorkspace(view.source);
          return {
            valid: true,
            error: "",
            viewId: root.props.viewId,
            nodes: nodes(root).map((node) => ({
              type: node.typeName,
              props: node.props,
            })),
            currentSummary: workspaceText(view.source),
          };
        } catch (error) {
          return { valid: false, error: String(error), nodes: [] };
        }
      },
    );
    return {
      path: path.slice(root.length + 1),
      session_id: result.session_id,
      prompt: result.prompt,
      complete: result.complete,
      assistant_text: result.assistant_text,
      views,
    };
  });
const output = resolve(process.argv[3] ?? join(root, "replay.json"));
writeFileSync(output, JSON.stringify(results, null, 2) + "\n");
console.log(
  JSON.stringify({
    turns: results.length,
    sources: results.reduce((n, result) => n + result.views.length, 0),
    invalid: results.flatMap((r) =>
      r.views
        .filter((v) => !v.valid)
        .map((v) => ({ path: r.path, error: v.error })),
    ),
    output,
  }),
);
