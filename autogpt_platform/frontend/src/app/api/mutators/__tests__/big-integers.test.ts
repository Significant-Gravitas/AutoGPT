import { http, HttpResponse } from "msw";
import { describe, expect, it } from "vitest";
import { server } from "@/mocks/mock-server";
import { customMutator } from "../custom-mutator";

const BIG = "9007199254740993";

function serveJson(method: "get" | "post", path: string, body: string) {
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
