import type { ExactInteger, JsonPath } from "@/lib/json-text";

/*
 * parseJsonExact hands a big integer to the app as its digit string, since a
 * JS number would round it. Sent back as is, a block input typed "any" would
 * turn from a number into a string. So each big integer read from a JSON
 * number is noted against the entity it belongs to (a graph, a review, a
 * preset, a schedule, a graph's run inputs) at its exact path (inside a graph,
 * starting with the node id), and only a
 * write to that same entity puts it back as a number, only at that path, and
 * only while the value there is still those digits. A string holding the same
 * digits anywhere else is left alone.
 *
 * Module state, so it lives in one browser tab only: the caller skips it on
 * the server, where it would be shared between users.
 */

type Json = Record<string, unknown>;

/** Values under `at` in a response belong to `entity`, at `prefix` + rest. */
interface ReadScope {
  entity: string;
  at: JsonPath;
  prefix?: JsonPath;
}

/**
 * Values under `at` in a request body may come from any of `entities`, from
 * the integers noted under `prefix` there.
 */
interface WriteScope {
  entities: string[];
  at: JsonPath;
  prefix?: JsonPath;
}

// Counts entities, so a graph is one entry however many nodes it has. In
// insertion order, oldest first; reading or writing an entity moves it last.
const MAX_ENTITIES = 200;
const integersByEntity = new Map<string, ExactInteger[]>();
// The entity of each kind ("graph", "review", ...) read or written last, so
// browsing other things never drops the graph or review that is open.
const latestOfKind = new Map<string, string>();

function touch(entity: string): void {
  const integers = integersByEntity.get(entity);
  integersByEntity.delete(entity);
  if (integers) integersByEntity.set(entity, integers);
  latestOfKind.set(entity.slice(0, entity.indexOf(":")), entity);
}

function evictOldest(): void {
  const latest = new Set(latestOfKind.values());
  for (const entity of integersByEntity.keys()) {
    if (integersByEntity.size <= MAX_ENTITIES) return;
    if (!latest.has(entity)) integersByEntity.delete(entity);
  }
}

function isRecord(value: unknown): value is Json {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function records(value: unknown): Json[] {
  return Array.isArray(value) ? value.filter(isRecord) : [];
}

function str(value: unknown): string | null {
  return typeof value === "string" && value !== "" ? value : null;
}

const runInputs = (graphId: string) => [
  `inputs:${graphId}`,
  `defaults:${graphId}`,
];

function graphScopes(graph: unknown, at: JsonPath = []): ReadScope[] {
  if (!isRecord(graph)) return [];
  const graphId = str(graph.id);
  if (!graphId) return [];
  return records(graph.nodes).flatMap((node, i) => {
    const nodeId = str(node.id);
    return nodeId
      ? [
          {
            entity: `graph:${graphId}`,
            at: [...at, "nodes", String(i)],
            prefix: [nodeId],
          },
        ]
      : [];
  });
}

function presetScope(preset: Json, at: JsonPath): ReadScope[] {
  const id = str(preset.id);
  return id ? [{ entity: `preset:${id}`, at: [...at, "inputs"] }] : [];
}

// Reviews come in lists, in a session, or inside the home page's attention
// items: wherever an object carries a node_exec_id and a payload.
function reviewScopes(value: unknown, at: string[] = []): ReadScope[] {
  if (Array.isArray(value)) {
    return value.flatMap((item, i) => reviewScopes(item, [...at, String(i)]));
  }
  if (!isRecord(value)) return [];
  const id = str(value.node_exec_id);
  if (id && "payload" in value) {
    return [{ entity: `review:${id}`, at: [...at, "payload"] }];
  }
  return Object.entries(value).flatMap(([key, child]) =>
    typeof child === "object" ? reviewScopes(child, [...at, key]) : [],
  );
}

const READS: [RegExp, (data: unknown) => ReadScope[]][] = [
  // A graph or one of its versions; also what a create or update returns.
  [/^\/api\/graphs(?:\/[^/]+(?:\/versions\/[^/]+)?)?$/, (d) => graphScopes(d)],
  // One execution: "Run again" sends its inputs back.
  [
    /^\/api\/(?:graphs\/[^/]+\/)?executions\/[^/]+$/,
    (d) =>
      isRecord(d) && str(d.graph_id) && "inputs" in d
        ? [{ entity: `inputs:${d.graph_id}`, at: ["inputs"] }]
        : [],
  ],
  // A library agent: its input defaults prefill the Run modal.
  [
    /^\/api\/library\/agents\/(?:by-graph\/)?[^/]+$/,
    (d) => {
      if (!isRecord(d) || !str(d.graph_id)) return [];
      const schema = d.input_schema;
      const properties = isRecord(schema) ? schema.properties : null;
      if (!isRecord(properties)) return [];
      return Object.keys(properties).map((name) => ({
        entity: `defaults:${d.graph_id}`,
        at: ["input_schema", "properties", name, "default"],
        prefix: [name],
      }));
    },
  ],
  [
    /^\/api\/library\/presets(?:\/[^/]+)?$/,
    (d) =>
      isRecord(d) && Array.isArray(d.presets)
        ? records(d.presets).flatMap((p, i) =>
            presetScope(p, ["presets", String(i)]),
          )
        : isRecord(d)
          ? presetScope(d, [])
          : [],
  ],
  [
    /^\/api\/(?:graphs\/[^/]+\/)?schedules$/,
    (d) =>
      Array.isArray(d)
        ? d.flatMap((job, i) => {
            const id = isRecord(job) ? str(job.id) : null;
            return id
              ? [{ entity: `schedule:${id}`, at: [String(i), "input_data"] }]
              : [];
          })
        : [],
  ],
  [/^\/api\/(?:review\/|home$)/, (d) => reviewScopes(d)],
];

const WRITES: [string, RegExp, (body: Json, id: string) => WriteScope[]][] = [
  [
    "PUT",
    /^\/api\/graphs\/([^/]+)$/,
    (body, graphId) =>
      records(body.nodes).map((node, i) => ({
        entities: [`graph:${graphId}`],
        at: ["nodes", String(i)],
        prefix: [String(node.id)],
      })),
  ],
  [
    "POST",
    /^\/api\/graphs()$/,
    (body) => {
      const graph = isRecord(body.graph) ? body.graph : {};
      const graphId = str(graph.id);
      if (!graphId) return [];
      return records(graph.nodes).map((node, i) => ({
        entities: [`graph:${graphId}`],
        at: ["graph", "nodes", String(i)],
        prefix: [String(node.id)],
      }));
    },
  ],
  [
    "POST",
    /^\/api\/review\/action()$/,
    (body) =>
      records(body.reviews).map((review, i) => ({
        entities: [`review:${String(review.node_exec_id)}`],
        at: ["reviews", String(i), "reviewed_data"],
      })),
  ],
  [
    "POST",
    /^\/api\/graphs\/([^/]+)\/(?:execute\/[^/]+|schedules)$/,
    (_, graphId) => [{ entities: runInputs(graphId), at: ["inputs"] }],
  ],
  [
    "POST",
    /^\/api\/library\/presets()$/,
    (body) => {
      const graphId = str(body.graph_id);
      return graphId ? [{ entities: runInputs(graphId), at: ["inputs"] }] : [];
    },
  ],
  [
    "PATCH",
    /^\/api\/library\/presets\/([^/]+)$/,
    (_, id) => [{ entities: [`preset:${id}`], at: ["inputs"] }],
  ],
  [
    "POST",
    /^\/api\/library\/presets\/([^/]+)\/execute$/,
    (_, id) => [{ entities: [`preset:${id}`], at: ["inputs"] }],
  ],
  [
    "PATCH",
    /^\/api\/schedules\/([^/]+)$/,
    (_, id) => [{ entities: [`schedule:${id}`], at: ["inputs"] }],
  ],
];

function routePath(url: string): string {
  const path = url.split(/[?#]/)[0];
  try {
    return decodeURIComponent(path);
  } catch {
    return path;
  }
}

function startsWith(path: JsonPath, prefix: JsonPath): boolean {
  return prefix.every((segment, i) => path[i] === segment);
}

/**
 * Notes the big integers of each entity in a response, replacing what an
 * earlier load of the same entity noted.
 */
export function recordIntegers(
  url: string,
  data: unknown,
  integers: ExactInteger[],
): void {
  const path = routePath(url);
  const read = READS.find(([route]) => route.test(path));
  if (!read) return;
  const scopes = read[1](data);
  for (const scope of scopes) {
    integersByEntity.delete(scope.entity);
    touch(scope.entity);
  }
  for (const { path: at, digits } of integers) {
    const scope = scopes.find((s) => startsWith(at, s.at));
    if (!scope) continue;
    const rest = at.slice(scope.at.length);
    const list = integersByEntity.get(scope.entity) ?? [];
    list.push({ path: [...(scope.prefix ?? []), ...rest], digits });
    integersByEntity.set(scope.entity, list);
  }
  evictOldest();
}

/**
 * True when `digits` at `path` inside `entity` were loaded as a JSON number,
 * e.g. a review payload: `isLoadedInteger("review:<id>", [], payload)`.
 */
export function isLoadedInteger(
  entity: string,
  path: JsonPath,
  digits: unknown,
): boolean {
  return (integersByEntity.get(entity) ?? []).some(
    (integer) =>
      integer.digits === digits &&
      integer.path.length === path.length &&
      startsWith(integer.path, path),
  );
}

function get(root: unknown, path: JsonPath): unknown {
  let node = root;
  for (const segment of path) {
    if (typeof node !== "object" || node === null) return undefined;
    node = (node as Json)[segment];
  }
  return node;
}

function set(root: unknown, path: JsonPath, value: unknown): void {
  const parent = get(root, path.slice(0, -1));
  if (typeof parent === "object" && parent !== null) {
    (parent as Json)[path[path.length - 1]] = value;
  }
}

/**
 * Puts big integers the same entity was loaded with back into a JSON request
 * body as numbers, where they are still the digits that were loaded. Any
 * other body, or one that is not JSON, comes back unchanged.
 */
export function restoreIntegers(
  method: string,
  url: string,
  body: string,
): string {
  if (integersByEntity.size === 0) return body;
  const path = routePath(url);
  const match = WRITES.map(
    ([m, route, scopes]) => [m === method && route.exec(path), scopes] as const,
  ).find(([found]) => found);
  if (!match) return body;
  const [found, scopesFor] = match;
  let parsed: unknown;
  try {
    parsed = JSON.parse(body);
  } catch {
    return body;
  }
  if (!isRecord(parsed) || !found) return body;
  let mark = "exact-integer:";
  while (body.includes(mark)) mark += "~";
  let restored = false;
  for (const scope of scopesFor(parsed, found[1] ?? "")) {
    const prefix = scope.prefix ?? [];
    for (const entity of scope.entities) {
      if (integersByEntity.has(entity)) touch(entity);
      for (const { path: at, digits } of integersByEntity.get(entity) ?? []) {
        if (!startsWith(at, prefix)) continue;
        const target = [...scope.at, ...at.slice(prefix.length)];
        if (target.length === 0 || get(parsed, target) !== digits) continue;
        set(parsed, target, mark + digits);
        restored = true;
      }
    }
  }
  if (!restored) return body;
  // The mark is letters, "-", ":" and "~": nothing a RegExp treats specially.
  return JSON.stringify(parsed).replace(
    new RegExp(`"${mark}(-?\\d+)"`, "g"),
    "$1",
  );
}
