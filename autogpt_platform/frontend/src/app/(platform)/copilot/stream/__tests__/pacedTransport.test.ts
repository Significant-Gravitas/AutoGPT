import { server } from "@/mocks/mock-server";
import { Chat } from "@ai-sdk/react";
import type { UIMessage } from "ai";
import { http, HttpResponse } from "msw";
import { afterEach, describe, expect, it, vi } from "vitest";

import { createCopilotTransport } from "../../copilotStreamTransport";
import { getShadowLog, resetShadows, setStreamShadowRate } from "../turnShadow";

const BASE = "http://localhost:18006";
vi.mock("@/services/environment", async (importActual) => {
  const actual = await importActual<typeof import("@/services/environment")>();
  return {
    ...actual,
    environment: { ...actual.environment, getAGPTServerBaseUrl: () => BASE },
  };
});
vi.mock("../../helpers", async (importActual) => {
  const actual = await importActual<typeof import("../../helpers")>();
  return { ...actual, getCopilotAuthHeaders: async () => ({}) };
});

afterEach(() => resetShadows());

// The SDK reads the teed branch through the word pacing, long after the tap
// has read the whole body; the turn must still end as a clean finish.
describe("a paced POST turn read through the shadow tee", () => {
  it(
    "finishes without a disconnect or an error",
    { timeout: 60_000 },
    async () => {
      setStreamShadowRate(1);
      server.use(
        http.post(
          `${BASE}/api/chat/sessions/s1/stream`,
          () =>
            new HttpResponse(arrivingOverTime(essayFrames(300), 5), {
              headers: { "content-type": "text/event-stream" },
            }),
        ),
      );
      const finishes: unknown[] = [];
      const errors: Error[] = [];
      const chat = new Chat<UIMessage>({
        id: "s1",
        transport: createCopilotTransport({
          sessionId: "s1",
          copilotModelRef: { current: undefined },
        }),
        onFinish: (args) => finishes.push(args),
        onError: (error) => errors.push(error),
      });

      await chat.sendMessage({ text: "Write an essay" });

      expect(errors).toEqual([]);
      expect(finishes).toEqual([
        expect.objectContaining({ isDisconnect: false, isError: false }),
      ]);
      expect(chat.status).toBe("ready");
      expect(getShadowLog("s1")?.status).toBe("finished");
    },
  );
});

function essayFrames(words: number) {
  let n = 0;
  const frame = (data: unknown) =>
    `id: t:${++n}-0\ndata: ${JSON.stringify(data)}\n\n`;
  const frames = [
    frame({ type: "start", messageId: "m1" }),
    frame({ type: "start-step" }),
    frame({ type: "text-start", id: "b" }),
  ];
  for (let i = 0; i < words; i += 6) {
    frames.push(
      frame({ type: "text-delta", id: "b", delta: "lighthouse ".repeat(6) }),
    );
  }
  frames.push(
    frame({ type: "text-end", id: "b" }),
    frame({ type: "finish-step" }),
    frame({ type: "finish" }),
    "data: [DONE]\n\n",
  );
  return frames;
}

function arrivingOverTime(frames: string[], everyMs: number) {
  const encoder = new TextEncoder();
  let i = 0;
  return new ReadableStream<Uint8Array>({
    async pull(controller) {
      if (i >= frames.length) {
        controller.close();
        return;
      }
      await new Promise((resolve) => setTimeout(resolve, everyMs));
      controller.enqueue(encoder.encode(frames[i++]));
    },
  });
}
