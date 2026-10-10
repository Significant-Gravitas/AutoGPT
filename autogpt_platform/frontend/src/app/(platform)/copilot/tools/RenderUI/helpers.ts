import type { RenderUIMessagePart } from "./isRenderUIPart";
import { z } from "zod/v4";
import { MAX_SOURCE_LENGTH } from "@/lib/openui/catalog";
import { parseWorkspace } from "@/lib/openui/parse";
import { asObject } from "../../components/ToolChain/resultHelpers";

const resultSchema = z.object({
  type: z.literal("ui_rendered"),
  version: z.literal(1),
  source: z.string().max(MAX_SOURCE_LENGTH),
  message: z.string().max(10_000),
  session_id: z.string().nullable().optional(),
});

export function readUIError(part: RenderUIMessagePart) {
  if (part.state === "output-error")
    return part.errorText || "The interactive view could not be prepared.";
  const output =
    part.state === "output-available" ? asObject(part.output) : null;
  if (output?.type !== "error") return null;
  return typeof output.message === "string" && output.message
    ? output.message
    : "The interactive view could not be prepared.";
}

export function readUIResult(part: RenderUIMessagePart) {
  const output =
    part.state === "output-available" ? asObject(part.output) : null;
  const parsed = resultSchema.safeParse(output);
  if (!parsed.success) {
    return {
      result: null,
      title: "Interactive view",
      valid: false,
      summary: typeof output?.message === "string" ? output.message : "",
    };
  }
  try {
    const root = parseWorkspace(parsed.data.source);
    return {
      result: parsed.data,
      title: String(root.props.title || "Interactive view"),
      valid: true,
      summary: parsed.data.message,
    };
  } catch {
    return {
      result: parsed.data,
      title: "Interactive view",
      valid: false,
      summary: parsed.data.message,
    };
  }
}

export function getStreamingSource(part: RenderUIMessagePart) {
  let input = asObject(part.input);
  if (
    part.type === "tool-run_capability" ||
    (part.type === "dynamic-tool" && part.toolName === "run_capability")
  )
    input = asObject(input?.input);
  return typeof input?.source === "string"
    ? input.source.slice(0, MAX_SOURCE_LENGTH)
    : "";
}

export function buildUIFollowUp(
  message: string,
  title: string,
  fields: Record<string, unknown>,
) {
  const values = Object.entries(fields)
    .slice(0, 24)
    .map(([name, value]) => [
      name.slice(0, 200),
      typeof value === "string" ? value.slice(0, 2000) : value,
    ]);
  const body = message.trim().slice(0, 2000);
  return `${body}\n\nFrom the interactive view: ${title.slice(0, 200)}${values.length ? `\nSubmitted values:\n${JSON.stringify(Object.fromEntries(values), null, 2)}` : ""}`;
}

export function readUIDraft(
  key: string,
  source: string,
): Record<string, unknown> {
  if (typeof window === "undefined") return {};
  try {
    const raw = sessionStorage.getItem(key);
    if (!raw || raw.length > 80_000) return {};
    const value = asObject(JSON.parse(raw));
    return value?.source === source ? (asObject(value.state) ?? {}) : {};
  } catch {
    return {};
  }
}

export function saveUIDraft(
  key: string,
  source: string,
  state: Record<string, unknown>,
) {
  try {
    const value = JSON.stringify({ source, state });
    if (value.length <= 80_000) sessionStorage.setItem(key, value);
  } catch {}
}
