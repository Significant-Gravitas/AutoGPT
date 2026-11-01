import { http, HttpResponse } from "msw";
import { beforeEach, describe, expect, it } from "vitest";
import {
  getV1GetSpecificGraph,
  putV1UpdateGraphVersion,
} from "@/app/api/__generated__/endpoints/graphs/graphs";
import type { BlockInfo } from "@/app/api/__generated__/models/blockInfo";
import type { GraphModel } from "@/app/api/__generated__/models/graphModel";
import { server } from "@/mocks/mock-server";
import { convertNodesPlusBlockInfoIntoCustomNodes } from "../components/helper";
import { useNodeStore } from "../stores/nodeStore";

const BIG = "9007199254740993";

// The backend stores the exact int; the body is written by hand because a
// JS object can't hold it.
const storedGraph = `{
  "id": "g1",
  "version": 1,
  "name": "Big ids",
  "description": "",
  "nodes": [{
    "id": "n1",
    "block_id": "b1",
    "input_default": {"value": ${BIG}, "records": [{"id": ${BIG}}], "count": 42, "label": "${BIG}"},
    "metadata": {"position": {"x": 0, "y": 0}}
  }],
  "links": []
}`;

describe("graph with an integer beyond 2^53", () => {
  let savedBody = "";

  beforeEach(() => {
    savedBody = "";
    useNodeStore.setState({ nodes: [] });
    server.use(
      http.get(
        "*/api/graphs/g1",
        () =>
          new HttpResponse(storedGraph, {
            headers: { "content-type": "application/json" },
          }),
      ),
      http.put("*/api/graphs/g1", async ({ request }) => {
        savedBody = await request.text();
        return new HttpResponse(savedBody, {
          headers: { "content-type": "application/json" },
        });
      }),
    );
  });

  it("loads the exact digits and saves them back unchanged", async () => {
    const loaded = (await getV1GetSpecificGraph("g1", { version: 1 }))
      .data as GraphModel;
    useNodeStore
      .getState()
      .setNodes(
        loaded.nodes!.map((node) =>
          convertNodesPlusBlockInfoIntoCustomNodes(node, {} as BlockInfo),
        ),
      );

    const saved = await putV1UpdateGraphVersion("g1", {
      id: loaded.id,
      name: "Renamed",
      description: loaded.description,
      nodes: useNodeStore.getState().getBackendNodes(),
      links: [],
    });

    // Numbers go back as numbers, so a field of any type keeps its type;
    // the string stays a string.
    expect(savedBody).toContain(
      `{"value":${BIG},"records":[{"id":${BIG}}],"count":42,"label":"${BIG}"}`,
    );
    expect(JSON.stringify(saved.data)).toContain(`"value":"${BIG}"`);
  });
});
