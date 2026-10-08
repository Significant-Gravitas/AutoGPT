import type { WireChunk } from "./turnConverter";
import type { TurnLog } from "./turnLog";

/**
 * Whether the AI SDK parser, which has seen exactly what `before` folded, can
 * take this chunk. It throws on a delta or end for a part it never saw start
 * and on an output for a call it never saw, and duplicates a part started
 * twice; the fold records those as protocol errors and the parser skips them.
 */
export function isRenderable(before: TurnLog, chunk: WireChunk): boolean {
  const id = typeof chunk.id === "string" ? chunk.id : "";
  const toolCallId =
    typeof chunk.toolCallId === "string" ? chunk.toolCallId : "";
  switch (chunk.type) {
    case "text-start":
    case "reasoning-start":
      return !before.blocks[id];
    case "text-delta":
    case "text-end":
      return before.blocks[id]?.kind === "text" && before.blocks[id].open;
    case "reasoning-delta":
    case "reasoning-end":
      return before.blocks[id]?.kind === "reasoning" && before.blocks[id].open;
    case "tool-input-start":
      return !before.tools[toolCallId];
    case "tool-input-delta":
    case "tool-output-available":
    case "tool-output-error":
      return !!before.tools[toolCallId];
    default:
      return true;
  }
}
