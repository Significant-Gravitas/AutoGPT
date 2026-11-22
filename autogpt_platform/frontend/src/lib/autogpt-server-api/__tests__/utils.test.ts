import type { BaseGraphInput } from "@/app/api/__generated__/models/baseGraphInput";
import type { Graph } from "@/app/api/__generated__/models/graph";
import type { Node } from "@/app/api/__generated__/models/node";
import { describe, expect, test } from "vitest";
import { sanitizeImportedGraph } from "../utils";

const AI_TEXT = "1f292d4a-41a4-4977-9684-7c8d560b9f91";
const CLAUDE_CODE = "4e34f4a5-9b89-4326-ba77-2dd6750b7194";
const FILL_TEMPLATE = "db7d8f02-2f44-4c55-ab7a-eae0941f0c30";
const CREATE_DICTIONARY = "b924ddf4-de4f-4b56-9a85-358930dcbc91";
const MCP_TOOL = "a0a4b1c2-d3e4-4f56-a7b8-c9d0e1f2a3b4";
const SUB_WORKFLOW = "e189baac-8c20-45a1-94a7-55177ea42565";
const AUTHENTICATED_WEB_REQUEST = "fff86bcd-e001-4bad-a7f6-2eae4720c8dc";
const SEND_EMAIL = "4335878a-394e-4e67-adf2-919877ff49ae";

const OPENAI_KEY = {
  id: "0b8f6c3e-openai",
  provider: "openai",
  type: "api_key",
  title: "Exporter's OpenAI key",
};
const E2B_KEY = { id: "5d1e9a72-e2b", provider: "e2b", type: "api_key" };
const ANTHROPIC_KEY = {
  id: "9c4b2f18-anthropic",
  provider: "anthropic",
  type: "api_key",
  title: null,
};
const MCP_LOGIN = {
  id: "77aa3c10-mcp",
  provider: "mcp",
  type: "oauth2",
  title: "Deploy server",
};

function node(
  id: string,
  blockID: string,
  inputs: Record<string, unknown>,
): Node {
  return {
    id,
    block_id: blockID,
    input_default: inputs,
    metadata: { position: { x: 0, y: 0 } },
    input_links: [],
    output_links: [],
  };
}

function workflow(nodes: Node[], subWorkflows: BaseGraphInput[] = []): Graph {
  return {
    name: "Support triage",
    description: "Answers and fixes support tickets",
    nodes,
    links: [],
    sub_graphs: subWorkflows,
  };
}

function importFile(graph: Graph): Graph {
  const parsed = JSON.parse(JSON.stringify(graph)) as Graph;
  sanitizeImportedGraph(parsed);
  return parsed;
}

function inputsOf(graph: Pick<Graph, "nodes">) {
  return graph.nodes?.map((n) => n.input_default);
}

describe("sanitizeImportedGraph", () => {
  test("removes each credentials input a node holds, `credentials` and `*_credentials` alike", () => {
    const imported = importFile(
      workflow([
        node("reply", AI_TEXT, {
          prompt: "Reply to {{ticket}}",
          model: "gpt-4o",
          credentials: OPENAI_KEY,
        }),
        node("fix", CLAUDE_CODE, {
          prompt: "Fix the failing test",
          e2b_credentials: E2B_KEY,
          anthropic_credentials: ANTHROPIC_KEY,
        }),
      ]),
    );

    expect(inputsOf(imported)).toEqual([
      { prompt: "Reply to {{ticket}}", model: "gpt-4o" },
      { prompt: "Fix the failing test" },
    ]);
  });

  test("removes a stored credential of any type, one this app doesn't know included", () => {
    const imported = importFile(
      workflow([
        node("lookup", AUTHENTICATED_WEB_REQUEST, {
          url: "https://tickets.example.com/api/42",
          credentials: {
            id: "3e7f2a90-http",
            provider: "http",
            type: "host_scoped",
            title: "Ticket API",
          },
        }),
        node("escalate", SEND_EMAIL, {
          to_email: "oncall@example.com",
          credentials: {
            id: "a91c4d07-smtp",
            provider: "smtp",
            type: "user_password",
          },
        }),
        node("reply", AI_TEXT, {
          prompt: "Reply to {{ticket}}",
          credentials: {
            id: "c5d2e6b1-openai",
            provider: "openai",
            type: "a_type_added_later",
          },
        }),
      ]),
    );

    expect(inputsOf(imported)).toEqual([
      { url: "https://tickets.example.com/api/42" },
      { to_email: "oncall@example.com" },
      { prompt: "Reply to {{ticket}}" },
    ]);
  });

  test("keeps a credentials input that names no stored credential", () => {
    const unpicked = { provider: "openai", type: "api_key" };
    const imported = importFile(
      workflow([
        node("reply", AI_TEXT, { prompt: "Reply", credentials: unpicked }),
      ]),
    );

    expect(inputsOf(imported)).toEqual([
      { prompt: "Reply", credentials: unpicked },
    ]);
  });

  test('keeps a template value, a dictionary entry and a tool argument keyed "credentials"', () => {
    const toolSchema = {
      type: "object",
      properties: { credentials: { type: "string" } },
      required: ["credentials"],
    };
    const imported = importFile(
      workflow([
        node("compose", FILL_TEMPLATE, {
          format: "Sign in as {{credentials}}",
          values: { credentials: "the on-call login" },
        }),
        node("payload", CREATE_DICTIONARY, {
          values: { credentials: { vault: "ops", path: "deploy/bot" } },
        }),
        node("deploy", MCP_TOOL, {
          server_url: "https://mcp.example.com/deploy",
          credentials: MCP_LOGIN,
          selected_tool: "deploy",
          tool_input_schema: toolSchema,
          tool_arguments: { credentials: "deploy-bot" },
        }),
      ]),
    );

    expect(inputsOf(imported)).toEqual([
      {
        format: "Sign in as {{credentials}}",
        values: { credentials: "the on-call login" },
      },
      { values: { credentials: { vault: "ops", path: "deploy/bot" } } },
      {
        server_url: "https://mcp.example.com/deploy",
        selected_tool: "deploy",
        tool_input_schema: toolSchema,
        tool_arguments: { credentials: "deploy-bot" },
      },
    ]);
  });

  test("removes the credentials inputs of an embedded sub-workflow's nodes", () => {
    const runTriage = {
      graph_id: "triage",
      graph_version: 3,
      user_id: "",
      inputs: { ticket: "#42" },
    };
    const imported = importFile(
      workflow(
        [node("triage", SUB_WORKFLOW, runTriage)],
        [
          {
            id: "triage",
            version: 3,
            name: "Triage",
            description: "Summarises and reproduces a ticket",
            nodes: [
              node("summarize", AI_TEXT, {
                prompt: "Summarize {{ticket}}",
                credentials: OPENAI_KEY,
              }),
              node("reproduce", CLAUDE_CODE, {
                prompt: "Reproduce {{ticket}}",
                e2b_credentials: E2B_KEY,
                anthropic_credentials: ANTHROPIC_KEY,
              }),
            ],
            links: [],
          },
        ],
      ),
    );

    expect(inputsOf(imported)).toEqual([runTriage]);
    expect(inputsOf(imported.sub_graphs![0])).toEqual([
      { prompt: "Summarize {{ticket}}" },
      { prompt: "Reproduce {{ticket}}" },
    ]);
  });

  test("keeps credentials-shaped data in an input that isn't a credentials field", () => {
    const record = { id: "record-1", provider: "openai", type: "api_key" };
    const imported = importFile(
      workflow([node("record", CREATE_DICTIONARY, { values: record })]),
    );

    expect(inputsOf(imported)).toEqual([{ values: record }]);
  });

  test("handles a file without sub-workflows or a node's inputs", () => {
    const imported = importFile({
      name: "Support triage",
      description: "Answers support tickets",
      nodes: [
        node("reply", AI_TEXT, { prompt: "Reply", credentials: OPENAI_KEY }),
        { id: "empty", block_id: CREATE_DICTIONARY },
      ],
      links: [],
    });

    expect(inputsOf(imported)).toEqual([{ prompt: "Reply" }, undefined]);
  });
});
