import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { waitFor } from "@testing-library/react";
import { http } from "msw";
import { server } from "@/mocks/mock-server";
import {
  assistantTextChunks,
  streamSseResponse,
} from "@/tests/integrations/copilot-sse";
import {
  getOrCreateCopilotChatRuntime,
  resetCopilotChatRegistry,
} from "../copilotChatRegistry";

vi.mock("@/services/environment", async (importActual) => {
  const actual = await importActual<typeof import("@/services/environment")>();
  return {
    ...actual,
    environment: {
      ...actual.environment,
      getAGPTServerBaseUrl: () => "http://localhost:18006",
    },
  };
});
vi.mock("../helpers", async (importActual) => {
  const actual = await importActual<typeof import("../helpers")>();
  return {
    ...actual,
    getCopilotAuthHeaders: async () => ({ "x-test-auth": "yes" }),
  };
});

beforeEach(resetCopilotChatRegistry);
afterEach(resetCopilotChatRegistry);

it("aborts a stalled GET and coalesces simultaneous resume requests before starting its replacement", async () => {
  let gets = 0;
  let firstAborted = false;
  const errors = vi.spyOn(console, "error");
  server.use(
    http.get(
      "http://localhost:18006/api/chat/sessions/get-resume/stream",
      ({ request }) => {
        gets++;
        if (gets === 1)
          request.signal.addEventListener("abort", () => {
            firstAborted = true;
          });
        return streamSseResponse(
          assistantTextChunks(gets === 1 ? "Stale" : "Recovered"),
          {
            abortSignal: request.signal,
            perChunkDelaysMs: gets === 1 ? [0, 0, 0, 120_000] : undefined,
          },
        );
      },
    ),
  );
  const runtime = getOrCreateCopilotChatRuntime("get-resume");
  const finishes = vi.fn();
  runtime.onFinish = finishes;
  const original = runtime.resumeStream();
  await waitFor(() => expect(runtime.chat.status).toBe("streaming"));
  const beforeResume = vi.fn();
  const replacement = runtime.resumeStream(beforeResume);
  expect(runtime.resumeStream(beforeResume)).toBe(replacement);
  await Promise.all([original, replacement]);
  expect(firstAborted).toBe(true);
  expect(gets).toBe(2);
  expect(beforeResume).toHaveBeenCalledTimes(1);
  expect(runtime.chat.status).toBe("ready");
  expect(runtime.chat.messages.at(-1)?.parts).toContainEqual({
    type: "text",
    text: "Recovered",
    state: "done",
  });
  expect(finishes.mock.calls.map(([args]) => args.isAbort)).toEqual([
    true,
    false,
  ]);
  expect(
    errors.mock.calls
      .flat()
      .some((error) => String(error).includes("reading 'state'")),
  ).toBe(false);
});

it("a user stop cancels a queued reconnect without issuing a GET", async () => {
  let gets = 0;
  let aborted = false;
  server.use(
    http.post(
      "http://localhost:18006/api/chat/sessions/stop-resume/stream",
      ({ request }) => {
        request.signal.addEventListener("abort", () => {
          aborted = true;
        });
        return streamSseResponse(assistantTextChunks("Should not arrive"), {
          abortSignal: request.signal,
          perChunkDelaysMs: [0, 0, 0, 120_000],
        });
      },
    ),
    http.get(
      "http://localhost:18006/api/chat/sessions/stop-resume/stream",
      () => {
        gets++;
        return streamSseResponse(assistantTextChunks("Should not resume"));
      },
    ),
  );
  const runtime = getOrCreateCopilotChatRuntime("stop-resume");
  const original = runtime.sendMessage({ text: "A long request" });
  await waitFor(() => expect(runtime.chat.status).toBe("streaming"));
  const beforeResume = vi.fn();
  const replacement = runtime.resumeStream(beforeResume);
  runtime.stop();
  await Promise.all([original, replacement]);
  await runtime.resumeStream(beforeResume);
  expect(aborted).toBe(true);
  expect(gets).toBe(0);
  expect(beforeResume).not.toHaveBeenCalled();
  expect(runtime.chat.status).toBe("ready");
});

it("finishes aborting before sending a new turn after Stop", async () => {
  const bodies: string[] = [];
  server.use(
    http.post(
      "http://localhost:18006/api/chat/sessions/send-after-stop/stream",
      async ({ request }) => {
        bodies.push(await request.text());
        return streamSseResponse(
          assistantTextChunks(
            bodies.length === 1 ? "Discarded answer" : "New answer",
          ),
          {
            abortSignal: request.signal,
            perChunkDelaysMs:
              bodies.length === 1 ? [0, 0, 0, 120_000] : undefined,
          },
        );
      },
    ),
  );
  const runtime = getOrCreateCopilotChatRuntime("send-after-stop");
  const original = runtime.sendMessage({ text: "Original request" });
  await waitFor(() => expect(runtime.chat.status).toBe("streaming"));
  runtime.stop();
  const next = runtime.sendMessage({ text: "Changed request" });
  await Promise.all([original, next]);
  expect(bodies).toHaveLength(2);
  expect(JSON.parse(bodies[0]).message).toBe("Original request");
  expect(JSON.parse(bodies[1]).message).toBe("Changed request");
  expect(runtime.chat.messages.at(-1)?.parts).toContainEqual({
    type: "text",
    text: "New answer",
    state: "done",
  });
  expect(runtime.chat.status).toBe("ready");
});
