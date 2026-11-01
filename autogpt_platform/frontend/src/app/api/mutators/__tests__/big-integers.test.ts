import { http, HttpResponse } from "msw";
import { describe, expect, it } from "vitest";
import { server } from "@/mocks/mock-server";
import { customMutator } from "../custom-mutator";

const BIG = "9007199254740993";

function serveJson(
  method: "get" | "post" | "put" | "patch",
  path: string,
  body: string,
) {
  let sent = "";
  server.use(
    http[method](`*${path}`, async ({ request }) => {
      sent = await request.text();
      return new HttpResponse(body, {
        headers: { "content-type": "application/json" },
      });
    }),
  );
  return () => sent;
}

async function send(method: string, path: string, body: unknown) {
  await customMutator(path, {
    method,
    body: typeof body === "string" ? body : JSON.stringify(body),
  });
}

async function load(path: string) {
  return (
    await customMutator<{ data: any; status: number; headers: Headers }>(path, {
      method: "GET",
    })
  ).data;
}

describe("big integers in loaded block data", () => {
  it.each([
    "/api/review/pending",
    "/api/review/execution/e1",
    "/api/library/agents/a1",
    "/api/library/presets/p1",
    "/api/store/listings/versions/v1/graph",
    "/api/executions/e1",
    "/api/graphs/g1/executions/e1",
    "/api/schedules",
    "/api/builder/search?search_query=x",
    "/api/home",
  ])("%s keeps the exact digits", async (path) => {
    serveJson("get", path, `{"value": ${BIG}, "count": 42}`);
    expect(await load(path)).toEqual({ value: BIG, count: 42 });
  });

  it("an unedited review is approved with the same JSON numbers", async () => {
    serveJson(
      "get",
      "/api/review/pending",
      `[{"node_exec_id": "n1", "payload": {"value": ${BIG}, "items": [${BIG}], "label": "${BIG}"}}]`,
    );
    const [review] = await load("/api/review/pending");
    const sent = serveJson("post", "/api/review/action", "{}");

    await customMutator("/api/review/action", {
      method: "POST",
      body: JSON.stringify({
        reviews: [
          { node_exec_id: review.node_exec_id, reviewed_data: review.payload },
        ],
      }),
    });

    expect(sent()).toContain(
      `"reviewed_data":{"value":${BIG},"items":[${BIG}],"label":"${BIG}"}`,
    );
  });

  it("digits typed as a string are sent as a string", async () => {
    const sent = serveJson("post", "/api/review/action", "{}");
    await customMutator("/api/review/action", {
      method: "POST",
      body: JSON.stringify({ note: BIG }),
    });
    expect(sent()).toBe(`{"note":"${BIG}"}`);
  });
});

// The graph the app loads: what it sends back to save it unchanged.
async function loadGraph(id: string, nodes: string) {
  serveJson("get", `/api/graphs/${id}`, `{"id": "${id}", "nodes": ${nodes}}`);
  return load(`/api/graphs/${id}`);
}

async function save(graph: { id: string }) {
  const sent = serveJson("put", `/api/graphs/${graph.id}`, "{}");
  await send("PUT", `/api/graphs/${graph.id}`, graph);
  return sent();
}

describe("big integers sent back to where they were loaded from", () => {
  it("an untouched graph save sends loaded numbers as JSON numbers", async () => {
    const graph = await loadGraph(
      "g-same",
      `[{"id": "n1", "input_default": {"value": ${BIG}, "records": [{"id": ${BIG}}], "label": "${BIG}"}}]`,
    );
    expect(await save(graph)).toBe(
      `{"id":"g-same","nodes":[{"id":"n1","input_default":{"value":${BIG},"records":[{"id":${BIG}}],"label":"${BIG}"}}]}`,
    );
  });

  it("a stored string beside a number under the same key stays a string", async () => {
    const graph = await loadGraph(
      "g-beside",
      `[{"id": "n1", "input_default": {"value": ${BIG}}}, {"id": "n2", "input_default": {"value": "${BIG}"}}]`,
    );
    expect(await save(graph)).toContain(
      `{"id":"n1","input_default":{"value":${BIG}}},{"id":"n2","input_default":{"value":"${BIG}"}}`,
    );
  });

  it("digits at the same array index of another field stay a string", async () => {
    const graph = await loadGraph(
      "g-index",
      `[{"id": "n1", "input_default": {"records": [${BIG}]}}]`,
    );
    graph.nodes[0].input_default = { tags: [BIG] };
    expect(await save(graph)).toContain(`"input_default":{"tags":["${BIG}"]}`);
  });

  it("digits edited into a field stay a string once it reloads without them", async () => {
    await loadGraph(
      "g-reload",
      `[{"id": "n1", "input_default": {"value": ${BIG}}}]`,
    );
    const graph = await loadGraph(
      "g-reload",
      `[{"id": "n1", "input_default": {"value": "${BIG}"}}]`,
    );
    expect(await save(graph)).toContain(`"value":"${BIG}"`);
  });

  it("a root-level string body is sent unchanged", async () => {
    serveJson("get", "/api/executions/e-root", BIG);
    expect(await load("/api/executions/e-root")).toBe(BIG);
    const sent = serveJson("post", "/api/review/action", "{}");
    await send("POST", "/api/review/action", JSON.stringify(BIG));
    expect(sent()).toBe(`"${BIG}"`);
  });

  it("a chat message after a graph load is sent unchanged", async () => {
    await loadGraph(
      "g-chat",
      `[{"id": "n1", "input_default": {"text": ${BIG}}}]`,
    );
    const sent = serveJson("post", "/api/chat/sessions/s1/stream", "{}");
    const body = JSON.stringify({ text: BIG, message: { text: BIG } });
    await send("POST", "/api/chat/sessions/s1/stream", body);
    expect(sent()).toBe(body);
  });

  it("a string of NUL and digits stays that string", async () => {
    serveJson(
      "get",
      "/api/review/execution/e-nul",
      `[{"node_exec_id": "r-nul", "payload": {"s": "\\u0000${BIG}", "n": ${BIG}}}]`,
    );
    const [review] = await load("/api/review/execution/e-nul");
    expect(review.payload).toEqual({ s: `\u0000${BIG}`, n: BIG });
    const sent = serveJson("post", "/api/review/action", "{}");
    await send("POST", "/api/review/action", {
      reviews: [{ node_exec_id: "r-nul", reviewed_data: review.payload }],
    });
    expect(sent()).toContain(`{"s":"\\u0000${BIG}","n":${BIG}}`);
  });

  it("a body that is not JSON is sent unchanged", async () => {
    await loadGraph("g-form", `[{"id": "n1", "input_default": {"x": ${BIG}}}]`);
    const sent = serveJson("put", "/api/graphs/g-form", "{}");
    await send("PUT", "/api/graphs/g-form", `card=${BIG}&x=1`);
    expect(sent()).toBe(`card=${BIG}&x=1`);
  });

  it("the same node id in another graph does not borrow the numbers", async () => {
    await loadGraph(
      "g-a",
      `[{"id": "n1", "input_default": {"value": ${BIG}}}]`,
    );
    const other = {
      id: "g-b",
      nodes: [{ id: "n1", input_default: { value: BIG } }],
    };
    expect(await save(other)).toContain(`"value":"${BIG}"`);
  });

  it("run again, presets and schedules send loaded inputs as numbers", async () => {
    serveJson(
      "get",
      "/api/executions/e-run",
      `{"id": "e-run", "graph_id": "g-run", "inputs": {"seed": ${BIG}}}`,
    );
    const run = await load("/api/executions/e-run");
    const executed = serveJson("post", "/api/graphs/g-run/execute/1", "{}");
    await send("POST", "/api/graphs/g-run/execute/1", { inputs: run.inputs });
    expect(executed()).toBe(`{"inputs":{"seed":${BIG}}}`);

    serveJson(
      "get",
      "/api/library/presets/p-run",
      `{"id": "p-run", "graph_id": "g-x", "inputs": {"seed": ${BIG}}}`,
    );
    const preset = await load("/api/library/presets/p-run");
    const patched = serveJson("patch", "/api/library/presets/p-run", "{}");
    await send("PATCH", "/api/library/presets/p-run", {
      inputs: preset.inputs,
    });
    expect(patched()).toBe(`{"inputs":{"seed":${BIG}}}`);

    serveJson(
      "get",
      "/api/schedules",
      `[{"id": "s-run", "graph_id": "g-x", "input_data": {"seed": ${BIG}}}]`,
    );
    const [schedule] = await load("/api/schedules");
    const updated = serveJson("patch", "/api/schedules/s-run", "{}");
    await send("PATCH", "/api/schedules/s-run", {
      inputs: schedule.input_data,
    });
    expect(updated()).toBe(`{"inputs":{"seed":${BIG}}}`);
  });

  it("a library agent's input defaults run as numbers", async () => {
    serveJson(
      "get",
      "/api/library/agents/a-run",
      `{"id": "a-run", "graph_id": "g-def", "input_schema": {"properties": {"seed": {"type": "integer", "default": ${BIG}}}}}`,
    );
    const agent = await load("/api/library/agents/a-run");
    const executed = serveJson("post", "/api/graphs/g-def/execute/1", "{}");
    await send("POST", "/api/graphs/g-def/execute/1", {
      inputs: { seed: agent.input_schema.properties.seed.default },
    });
    expect(executed()).toBe(`{"inputs":{"seed":${BIG}}}`);
  });
});
