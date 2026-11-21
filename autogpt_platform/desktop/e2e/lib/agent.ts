// Create an agent and run it through the same doors the app's own pages
// use: the REST API behind /api/proxy and the websocket at /_agpt/ws.
//
// Routes: backend/api/features/graphs/routes.py (create, execute),
// graph_executions/routes.py (result), ws_api.py (events).

import { randomUUID } from "node:crypto";

import type { Page } from "@playwright/test";

import { call, type Reply } from "./session";

// CalculatorBlock (backend/blocks/maths.py): needs no credentials, no network
// and no model, so the run exercises the executor and nothing else.
const CALCULATOR_BLOCK = "b1ab9b19-67a6-406d-abf5-2dba76d00c79";

export interface Graph {
  id: string;
  version: number;
}

export interface Run {
  /** COMPLETED, FAILED, TERMINATED, or why no final event arrived. */
  outcome: string;
  executionId: string | null;
  /** The last websocket messages, for a failure message. */
  messages: unknown[];
}

/** One node: 10 + 5. */
export async function createCalculatorAgent(page: Page, name: string): Promise<Reply> {
  return call(page, "POST", "/api/proxy/api/graphs", {
    graph: {
      name,
      description: "Created by the desktop end-to-end test",
      nodes: [
        {
          id: randomUUID(),
          block_id: CALCULATOR_BLOCK,
          input_default: { operation: "Add", a: 10, b: 5 },
          metadata: { position: { x: 0, y: 0 } },
        },
      ],
      links: [],
    },
  });
}

/** Run the graph and wait for its final event on the websocket.
 *
 * The page subscribes to the graph before starting the run, so a run that
 * finishes at once cannot be missed. The token is the one the app's own
 * websocket client uses (Better Auth's /api/auth/token). */
export async function runAndWatch(page: Page, graph: Graph, timeoutMs: number): Promise<Run> {
  return page.evaluate(
    async ({ graphId, version, timeoutMs }) => {
      const { token } = await (await fetch("/api/auth/token")).json();
      const address = `${location.origin.replace(/^http/, "ws")}/_agpt/ws?token=${encodeURIComponent(token)}`;
      const socket = new WebSocket(address);
      const messages: any[] = [];
      let executionId: string | null = null;

      return new Promise<{ outcome: string; executionId: string | null; messages: unknown[] }>((resolve) => {
        function finish(outcome: string) {
          clearTimeout(timer);
          resolve({ outcome, executionId, messages: messages.slice(-15) });
          socket.close();
        }
        function finalStatus(): string | undefined {
          return messages
            .filter((message) => message.method === "graph_execution_event" && message.data?.id === executionId)
            .map((message) => message.data.status)
            .find((status) => ["COMPLETED", "FAILED", "TERMINATED"].includes(status));
        }
        async function execute() {
          const response = await fetch(`/api/proxy/api/graphs/${graphId}/execute/${version}`, {
            method: "POST",
            headers: { "content-type": "application/json" },
            body: JSON.stringify({ inputs: {}, credentials_inputs: {} }),
          });
          if (!response.ok) return finish(`execute returned ${response.status}: ${await response.text()}`);
          executionId = (await response.json()).id;
        }

        const timer = setTimeout(() => finish(`no final event within ${timeoutMs / 1000}s`), timeoutMs);
        socket.onerror = () => finish("the websocket failed");
        socket.onclose = (event) => finish(`the websocket closed (${event.code} ${event.reason})`);
        socket.onopen = () =>
          socket.send(JSON.stringify({ method: "subscribe_graph_executions", data: { graph_id: graphId } }));
        socket.onmessage = async (event) => {
          const message = JSON.parse(event.data);
          messages.push(message);
          if (message.method === "subscribe_graph_executions") {
            if (!message.success) return finish(`subscribing failed: ${event.data}`);
            await execute();
          }
          const status = executionId && finalStatus();
          if (status) finish(status);
        };
      });
    },
    { graphId: graph.id, version: graph.version, timeoutMs },
  );
}

export async function executionResult(page: Page, graphId: string, executionId: string): Promise<Reply> {
  return call(page, "GET", `/api/proxy/api/graphs/${graphId}/executions/${executionId}`);
}
