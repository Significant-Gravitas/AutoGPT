import { z } from "zod/v4";
import { MAX_SOURCE_LENGTH } from "@/lib/openui/catalog";

const streamEvent = z.discriminatedUnion("type", [
  z.object({ type: z.literal("delta"), text: z.string() }),
  z.object({ type: z.literal("done") }),
  z.object({ type: z.literal("error"), message: z.string() }),
]);

export function getActionFields(state: unknown, formName?: string) {
  if (!state || typeof state !== "object") return {};
  const record = state as Record<string, unknown>;
  const fields = formName ? record[formName] : record;
  if (!fields || typeof fields !== "object") return {};
  const result: Record<string, string | number | boolean> = {};
  for (const [name, field] of Object.entries(fields)) {
    const value =
      field && typeof field === "object" && "value" in field
        ? field.value
        : field;
    if (
      typeof value === "string" ||
      typeof value === "number" ||
      typeof value === "boolean"
    )
      result[name] = value;
  }
  return result;
}

export async function streamLiveWorkspace(
  input: { prompt: string; source: string; fields: Record<string, unknown> },
  signal: AbortSignal,
  update: (source: string) => void,
) {
  const response = await fetch("/api/openui", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(input),
    signal,
  });
  if (!response.ok) {
    const body: unknown = await response.json().catch(() => null);
    throw new Error(
      body &&
      typeof body === "object" &&
      "error" in body &&
      typeof body.error === "string"
        ? body.error
        : "Couldn't connect to live generation. Please try again.",
    );
  }
  if (!response.body)
    throw new Error("No response received. Please try again.");
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let source = "";
  let completed = false;
  try {
    while (true) {
      const { value, done } = await reader.read();
      buffer += decoder.decode(value, { stream: !done });
      const lines = buffer.split("\n");
      buffer = lines.pop() ?? "";
      for (const line of lines.filter(Boolean)) {
        const event = streamEvent.parse(JSON.parse(line));
        if (event.type === "error") throw new Error(event.message);
        if (event.type === "done") completed = true;
        if (event.type === "delta") {
          source += event.text;
          if (source.length > MAX_SOURCE_LENGTH)
            throw new Error(
              "The response is too large. Try a more focused request.",
            );
          update(source);
        }
      }
      if (done) break;
      if (buffer.length > MAX_SOURCE_LENGTH)
        throw new Error("Invalid response from generation service.");
    }
    if (!completed)
      throw new Error(
        "The connection ended before the workspace was ready. Please try again.",
      );
    return source;
  } finally {
    await reader.cancel().catch(() => {});
    reader.releaseLock();
  }
}

export async function streamSample(
  source: string,
  signal: AbortSignal,
  update: (source: string) => void,
) {
  for (let end = 0; end < source.length; end += 100) {
    signal.throwIfAborted();
    update(source.slice(0, end + 100));
    await new Promise((resolve) => setTimeout(resolve, 18));
  }
  signal.throwIfAborted();
  return source;
}

export function exportWorkspace(source: string) {
  const url = URL.createObjectURL(
    new Blob([source], { type: "text/plain;charset=utf-8" }),
  );
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = "autogpt-workspace.openui";
  anchor.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
