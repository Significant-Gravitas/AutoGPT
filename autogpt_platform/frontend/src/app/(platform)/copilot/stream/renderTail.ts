import type { UIDataTypes, UIMessage, UITools } from "ai";

import {
  convertChatSessionMessagesToUiMessages,
  createTurnLogRenderer,
  type TurnStatsMap,
} from "../helpers/convertChatSessionToUiMessages";
import type { RowsSegment, RuntimeSnapshot } from "./turnRuntime";

type UiMessage = UIMessage<unknown, UIDataTypes, UITools>;

/**
 * Render the runtime's tail with one renderer per turn, so a turn's bubbles
 * keep their objects, and so their keys, for the life of the mount.
 */
export function createTailRenderer(sessionId: string) {
  const turnRenderers = new Map<
    string,
    ReturnType<typeof createTurnLogRenderer>
  >();
  const rowsMessages = new WeakMap<RowsSegment, UiMessage[]>();

  return function renderTail(snapshot: RuntimeSnapshot) {
    const messages: UiMessage[] = [];
    const stats: TurnStatsMap = new Map();
    const drained = drainedTexts(snapshot);
    for (const seg of snapshot.segments) {
      if (seg.kind === "user") {
        // A drained row in the turn stands in for its promoted chip.
        if (seg.origin === "chip" && takeOne(drained, textOf(seg.message))) {
          continue;
        }
        messages.push(seg.message as UiMessage);
        if (seg.rawId) stats.set(seg.key, { rawMessageId: seg.rawId });
      } else if (seg.kind === "rows") {
        let converted = rowsMessages.get(seg);
        if (!converted) {
          converted = convertChatSessionMessagesToUiMessages(
            sessionId,
            [...seg.rows],
            { isComplete: true },
          ).messages;
          rowsMessages.set(seg, converted);
        }
        messages.push(...converted);
      } else {
        let render = turnRenderers.get(seg.key);
        if (!render) {
          render = createTurnLogRenderer();
          turnRenderers.set(seg.key, render);
        }
        const rendered = render(seg.log);
        messages.push(...rendered);
        const last = rendered.findLast((m) => m.role === "assistant");
        if (last && (seg.durationMs !== null || seg.createdAt)) {
          stats.set(last.id, {
            ...(seg.durationMs !== null ? { durationMs: seg.durationMs } : {}),
            ...(seg.createdAt ? { createdAt: seg.createdAt } : {}),
          });
        }
      }
    }
    return { messages, stats };
  };
}

function drainedTexts(snapshot: RuntimeSnapshot) {
  const texts = new Map<string, number>();
  for (const seg of snapshot.segments) {
    if (seg.kind !== "turn") continue;
    for (const row of seg.log.rows) {
      if (row.role !== "user") continue;
      texts.set(row.content, (texts.get(row.content) ?? 0) + 1);
    }
  }
  return texts;
}

function takeOne(texts: Map<string, number>, text: string) {
  const count = texts.get(text) ?? 0;
  if (count === 0) return false;
  texts.set(text, count - 1);
  return true;
}

function textOf(message: UIMessage) {
  return message.parts
    .map((part) => (part.type === "text" ? part.text : ""))
    .join("");
}
