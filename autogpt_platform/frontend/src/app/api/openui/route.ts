import { createOpenAICompatible } from "@ai-sdk/openai-compatible";
import { streamText } from "ai";
import { getServerAuthToken } from "@/lib/auth/server/getServerAuthToken";
import { getSystemPrompt, MAX_SOURCE_LENGTH } from "@/lib/openui/catalog";
import { getOpenUIConfig, isOpenUIEnabled } from "@/lib/openui/config";
import { readRequest } from "./helpers";

export const maxDuration = 60;

export async function POST(request: Request) {
  if (!isOpenUIEnabled())
    return Response.json({ error: "Not found" }, { status: 404 });
  const origin = request.headers.get("origin");
  if (origin && origin !== new URL(request.url).origin)
    return Response.json({ error: "Invalid origin" }, { status: 403 });
  if (!(await getServerAuthToken()))
    return Response.json(
      { error: "Sign in to AutoGPT to use live generation." },
      { status: 401 },
    );
  let input;
  try {
    input = await readRequest(request);
  } catch {
    return Response.json(
      { error: "Please send a shorter prompt and a valid workspace." },
      { status: 400 },
    );
  }
  const config = getOpenUIConfig();
  if (!config)
    return Response.json(
      {
        error:
          "Live AI is not configured in this environment. You can explore the prepared examples in Sample mode.",
      },
      { status: 503 },
    );
  const abort = new AbortController();
  const provider = createOpenAICompatible({
    name: "openui-experiment",
    baseURL: config.baseURL,
    apiKey: config.apiKey,
  });
  const result = streamText({
    model: provider(config.model),
    system: getSystemPrompt(),
    messages: [
      {
        role: "user",
        content: JSON.stringify({
          request: input.prompt,
          currentWorkspace: input.source,
          submittedFields: input.fields,
        }),
      },
    ],
    maxOutputTokens: 6000,
    maxRetries: 0,
    abortSignal: AbortSignal.any([
      request.signal,
      abort.signal,
      AbortSignal.timeout(50_000),
    ]),
  });
  const encoder = new TextEncoder();
  let cancelled = false;
  const stream = new ReadableStream({
    async start(controller) {
      let length = 0;
      let finished = false;
      function emit(event: object) {
        if (!cancelled)
          controller.enqueue(encoder.encode(JSON.stringify(event) + "\n"));
      }
      try {
        for await (const part of result.fullStream) {
          if (part.type === "error") throw new Error("Provider error");
          if (part.type === "text-delta") {
            length += part.text.length;
            if (length > MAX_SOURCE_LENGTH) throw new Error("Output too large");
            emit({ type: "delta", text: part.text });
          }
          if (part.type === "finish") {
            if (part.finishReason !== "stop")
              throw new Error("Incomplete generation");
            finished = true;
          }
        }
        if (
          !finished ||
          !length ||
          abort.signal.aborted ||
          request.signal.aborted
        )
          throw new Error("Incomplete generation");
        emit({ type: "done" });
      } catch {
        abort.abort();
        if (!request.signal.aborted)
          emit({
            type: "error",
            message:
              "Live generation could not finish. Try again, or switch to a sample.",
          });
      } finally {
        if (!cancelled) controller.close();
      }
    },
    cancel() {
      cancelled = true;
      abort.abort();
    },
  });
  return new Response(stream, {
    headers: {
      "Content-Type": "application/x-ndjson",
      "Cache-Control": "no-store",
      "X-Accel-Buffering": "no",
    },
  });
}
