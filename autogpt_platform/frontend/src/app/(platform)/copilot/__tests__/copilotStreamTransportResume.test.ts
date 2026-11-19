import { server } from "@/mocks/mock-server";
import type { UIMessageChunk } from "ai";
import { http, HttpResponse } from "msw";
import { describe, expect, it, vi } from "vitest";

import { createCopilotTransport } from "../copilotStreamTransport";

const BASE = "http://localhost:18006";
vi.mock("@/services/environment", async (importActual) => {
  const actual = await importActual<typeof import("@/services/environment")>();
  return {
    ...actual,
    environment: { ...actual.environment, getAGPTServerBaseUrl: () => BASE },
  };
});
vi.mock("../helpers", async (importActual) => {
  const actual = await importActual<typeof import("../helpers")>();
  return { ...actual, getCopilotAuthHeaders: async () => ({}) };
});

const REPLAY: UIMessageChunk[] = [
  { type: "start", messageId: "m1" },
  { type: "text-start", id: "b" },
  { type: "text-delta", id: "b", delta: "Already written before the reload. " },
  { type: "text-delta", id: "b", delta: "More history here. " },
  { type: "text-end", id: "b" },
  { type: "finish" },
];

function sse(chunks: UIMessageChunk[]) {
  return [
    ...chunks.map((chunk) => `data: ${JSON.stringify(chunk)}\n\n`),
    "data: [DONE]\n\n",
  ].join("");
}

describe("resuming a running turn", () => {
  it("shows the replayed history as it was sent, unsplit", async () => {
    server.use(
      http.get(
        `${BASE}/api/chat/sessions/s1/stream`,
        () =>
          new HttpResponse(sse(REPLAY), {
            headers: { "content-type": "text/event-stream" },
          }),
      ),
    );
    const transport = createCopilotTransport({
      sessionId: "s1",
      copilotModelRef: { current: undefined },
    });

    const stream = await transport.reconnectToStream({ chatId: "s1" });
    const out: UIMessageChunk[] = [];
    const reader = stream!.getReader();
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      out.push(value);
    }

    expect(out).toEqual(REPLAY);
  });
});
