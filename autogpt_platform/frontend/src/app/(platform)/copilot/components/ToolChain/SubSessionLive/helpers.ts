import type { SessionDetailResponse } from "@/app/api/__generated__/models/sessionDetailResponse";
import type { ToolUIPart } from "ai";
import {
  getAnimationText,
  getToolCategory,
} from "../../../tools/GenericTool/helpers";
import type { ChainRow } from "../helpers";
import { getCatalogLabel } from "../toolCatalog";

export const POLL_MS = 3000;
// Live polling stops after this long on screen — a marathon sub-session is
// better followed via its "Open sub-session" link than by pinging the API
// every few seconds indefinitely. The card keeps its last snapshot.
export const POLL_CAP_MS = 5 * 60_000;

export interface LiveStep {
  name: string;
  input: unknown;
}

export function isSessionLive(session: SessionDetailResponse): boolean {
  if (session.active_stream) return true;
  const status = session.chat_status?.toLowerCase();
  return status === "running" || status === "queued";
}

export function getLiveNotice({
  isError,
  isPaused,
}: {
  isError: boolean;
  isPaused: boolean;
}) {
  if (isError) return "Couldn't load live updates";
  if (isPaused) return "Live updates paused";
  return null;
}

/** A re-delegation reuses the same sub-session, so only the CURRENT turn
 *  (everything after the last user message) belongs to this card — the full
 *  history would replay the previous delegation's final answer here. */
export function collectCurrentTurn(session: SessionDetailResponse) {
  const allMessages = Array.isArray(session.messages) ? session.messages : [];
  const lastUserIndex = allMessages.findLastIndex((m) => m.role === "user");
  const messages =
    lastUserIndex === -1 ? allMessages : allMessages.slice(lastUserIndex + 1);
  const steps: LiveStep[] = [];
  let latestText: string | null = null;
  for (const msg of messages) {
    const toolCalls = Array.isArray(msg.tool_calls) ? msg.tool_calls : [];
    for (const rawCall of toolCalls) {
      if (!rawCall || typeof rawCall !== "object") continue;
      const call = rawCall as {
        function?: { name?: unknown; arguments?: unknown };
      };
      const name = String(call.function?.name ?? "").trim();
      if (name)
        steps.push({ name, input: parseArguments(call.function?.arguments) });
    }
    if (
      msg.role === "assistant" &&
      typeof msg.content === "string" &&
      msg.content.trim()
    ) {
      latestText = msg.content.trim();
    }
  }
  return { steps, latestText };
}

/** Dress a polled tool call as a ChainRow so the delegate's steps reuse the
 *  main chain's icons and labels instead of raw tool names. */
export function toMiniRow(
  step: LiveStep,
  index: number,
  running: boolean,
): ChainRow {
  const state = running ? "running" : "done";
  const catalog = getCatalogLabel(step.name, step.input, state);
  const category = catalog?.category ?? getToolCategory(step.name);
  const text =
    catalog?.text ??
    getAnimationText(
      {
        type: `tool-${step.name}`,
        state: running ? "input-available" : "output-available",
        input: step.input,
        toolCallId: `live-${index}`,
      } as ToolUIPart,
      getToolCategory(step.name),
    );
  return {
    key: `live-${step.name}-${index}`,
    category,
    text,
    state,
    tool: step.name,
    input: step.input,
  };
}

export function parseArguments(rawArguments: unknown): unknown {
  if (typeof rawArguments !== "string") return rawArguments ?? {};
  try {
    return JSON.parse(rawArguments);
  } catch {
    return {};
  }
}
