import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { z } from "zod/v4";
import { parseWorkspace } from "../src/lib/openui/parse";

const caseSchema = z.object({
  id: z.string(),
  title: z.string(),
  category: z.string(),
  prompt: z.string(),
  ui_expectation: z.enum(["required", "forbidden", "optional"]),
  component_groups: z.array(z.array(z.string())),
});
const resultSchema = z.object({
  session_id: z.string(),
  elapsed_seconds: z.number(),
  assistant_text: z.string().optional(),
  error: z.unknown().optional(),
  stream_errors: z.array(z.unknown()).optional(),
  stream_completed: z.boolean().optional(),
  chat_status: z.string().optional(),
  views: z
    .array(z.object({ source: z.string(), summary: z.string() }))
    .default([]),
});

interface Node {
  type: string;
  props: Record<string, unknown>;
}

function collectNodes(value: unknown): Node[] {
  if (Array.isArray(value)) return value.flatMap(collectNodes);
  if (!value || typeof value !== "object") return [];
  const nodes = Object.values(value).flatMap(collectNodes);
  if (
    "typeName" in value &&
    typeof value.typeName === "string" &&
    "props" in value &&
    value.props &&
    typeof value.props === "object"
  )
    nodes.unshift({
      type: value.typeName,
      props: value.props as Record<string, unknown>,
    });
  return nodes;
}

function inspectSource(source: string) {
  try {
    const nodes = collectNodes(parseWorkspace(source));
    return { valid: true, error: null, nodes };
  } catch (error) {
    return {
      valid: false,
      error: error instanceof Error ? error.message : String(error),
      nodes: [] as Node[],
    };
  }
}

function grade(test: z.infer<typeof caseSchema>, path: string) {
  if (!existsSync(path)) return { ...test, status: "pending" };
  const result = resultSchema.parse(JSON.parse(readFileSync(path, "utf8")));
  const views = result.views.map((view) => ({
    ...view,
    ...inspectSource(view.source),
  }));
  const components = [
    ...new Set(views.flatMap((view) => view.nodes.map((node) => node.type))),
  ];
  const missing = test.component_groups.filter(
    (group) => !group.some((type) => components.includes(type)),
  );
  const surfaced =
    test.ui_expectation === "optional" ||
    (test.ui_expectation === "required"
      ? views.length > 0
      : views.length === 0);
  const valid = views.every((view) => view.valid);
  const complete =
    !result.error &&
    !result.stream_errors?.length &&
    result.stream_completed === true &&
    result.chat_status === "idle";
  return {
    ...test,
    session_id: result.session_id,
    elapsed_seconds: result.elapsed_seconds,
    status: complete && surfaced && valid && !missing.length ? "pass" : "fail",
    complete,
    surfaced,
    valid,
    components,
    missing_component_groups: missing,
    error: result.error ?? null,
    stream_errors: result.stream_errors ?? [],
    assistant_text: result.assistant_text ?? "",
    views,
  };
}

const directory = resolve(process.argv[2] ?? ".");
const run = process.argv[3] ?? "baseline";
const cases = readFileSync(join(directory, "cases.jsonl"), "utf8")
  .trim()
  .split("\n")
  .map((line) => caseSchema.parse(JSON.parse(line)));
const results = cases.map((test) =>
  grade(test, join(directory, "runs", run, "cases", `${test.id}.json`)),
);
const report = {
  checked_at: new Date().toISOString(),
  run,
  rubric:
    "Automated only: completed live turn, expected UI selection, canonical parser/schema validity, and task-appropriate component groups. A pass does not certify factual correctness, visual quality, or interaction behavior.",
  totals: Object.fromEntries(
    ["pass", "fail", "pending"].map((status) => [
      status,
      results.filter((result) => result.status === status).length,
    ]),
  ),
  results,
};
writeFileSync(
  join(directory, "runs", run, "grades.json"),
  JSON.stringify(report, null, 2),
);
console.log(
  JSON.stringify({
    ...report.totals,
    failures: results
      .filter((result) => result.status === "fail")
      .map((result) => ({
        id: result.id,
        ...("missing_component_groups" in result
          ? {
              missing: result.missing_component_groups,
              complete: result.complete,
              surfaced: result.surfaced,
              valid: result.valid,
              error: result.error,
            }
          : {}),
      })),
  }),
);
