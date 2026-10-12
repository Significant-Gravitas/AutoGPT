import {
  parseJsonEventStream,
  readUIMessageStream,
  uiMessageChunkSchema,
  type UIMessage,
  type UIMessageChunk,
} from "ai";
import { describe, expect, it } from "vitest";
import { loadRecordedTurn } from "./backend-sim";

// The backend's resume ids and checkpoints ship before the client reads them,
// so the AI SDK parser the chat runs today must build the same message
// without them.
describe("the recorded turns through the AI SDK's own parser", () => {
  it.each([
    "dummy-text-turn",
    "baseline-tool-turn",
    "sdk-late-tool-result",
    "baseline-drain-turn",
    "sdk-reasoning-turn",
    "sdk-auto-continue-turn",
    "baseline-consecutive-tools-turn",
    "sdk-consecutive-tools-turn",
  ])("%s: entry ids and the checkpoint change nothing", async (name) => {
    const { frames } = loadRecordedTurn(name);
    const bare = frames
      .filter(({ sse }) => !sse.includes('"type": "data-checkpoint"'))
      .map(({ sse }) => sse.replace(/^id: .*\n/, ""));

    expect(bare.length).toBeLessThan(frames.length);
    const dataFrames = frames.filter(({ sse }) => !sse.startsWith(":"));
    expect(dataFrames.every(({ sse }) => sse.startsWith("id: "))).toBe(true);
    expect(await finalMessage(frames.map(({ sse }) => sse))).toEqual(
      await finalMessage(bare),
    );
  });
});

async function finalMessage(sse: string[]) {
  const body = new Response(sse.join("") + "data: [DONE]\n\n").body!;
  const chunks = parseJsonEventStream({
    stream: body,
    schema: uiMessageChunkSchema,
  }).pipeThrough(
    new TransformStream({
      transform(parsed, controller) {
        if (!parsed.success) throw parsed.error;
        controller.enqueue(parsed.value as UIMessageChunk);
      },
    }),
  );
  let last: UIMessage | undefined;
  for await (const message of readUIMessageStream({ stream: chunks })) {
    last = message;
  }
  return last;
}
