export const SSE_HEADERS = {
  "Content-Type": "text/event-stream",
  "Cache-Control": "no-cache, no-transform",
  Connection: "keep-alive",
  "X-Accel-Buffering": "no",
} as const;

export function normalizeSSEStream(
  input: ReadableStream<Uint8Array>,
): ReadableStream<Uint8Array> {
  const decoder = new TextDecoder();
  const encoder = new TextEncoder();
  let buffer = "";

  return input.pipeThrough(
    new TransformStream<Uint8Array, Uint8Array>({
      transform(chunk, controller) {
        buffer += decoder.decode(chunk, { stream: true });

        const parts = buffer.split("\n\n");
        buffer = parts.pop() ?? "";

        for (const part of parts) {
          const normalized = normalizeSSEEvent(part);
          controller.enqueue(encoder.encode(normalized + "\n\n"));
        }
      },
      flush(controller) {
        if (buffer.trim()) {
          const normalized = normalizeSSEEvent(buffer);
          controller.enqueue(encoder.encode(normalized + "\n\n"));
        }
      },
    }),
  );
}

function normalizeSSEEvent(event: string): string {
  const lines = event.split("\n");
  const dataLines: string[] = [];
  const otherLines: string[] = [];

  for (const line of lines) {
    if (line.startsWith("data: ")) {
      dataLines.push(line.slice(6));
    } else {
      otherLines.push(line);
    }
  }

  if (dataLines.length === 0) return event;

  const dataStr = dataLines.join("\n");
  try {
    const parsed = JSON.parse(dataStr) as Record<string, unknown>;
    if (parsed.type === "error") {
      const normalized = {
        type: "error",
        errorText:
          typeof parsed.errorText === "string"
            ? parsed.errorText
            : "An unexpected error occurred",
      };
      const newData = `data: ${JSON.stringify(normalized)}`;
      return [...otherLines.filter((l) => l.length > 0), newData].join("\n");
    }
  } catch {
    // Not valid JSON — pass through as-is
  }

  return event;
}
