import { z } from "zod/v4";
import { MAX_SOURCE_LENGTH } from "@/lib/openui/catalog";

export const generationRequest = z.object({
  prompt: z.string().trim().min(1).max(4000),
  source: z.string().max(MAX_SOURCE_LENGTH).default(""),
  fields: z
    .record(
      z.string().max(100),
      z.union([z.string().max(500), z.number(), z.boolean()]),
    )
    .refine((fields) => Object.keys(fields).length <= 10)
    .default({}),
});

export async function readRequest(request: Request) {
  const reader = request.body?.getReader();
  if (!reader) throw new Error("Missing request body");
  const decoder = new TextDecoder();
  let size = 0;
  let body = "";
  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      size += value.byteLength;
      if (size > 80_000) {
        await reader.cancel();
        throw new Error("Request too large");
      }
      body += decoder.decode(value, { stream: true });
    }
    return generationRequest.parse(JSON.parse(body + decoder.decode()));
  } finally {
    reader.releaseLock();
  }
}
