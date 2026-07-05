import type { UIMessage } from "ai";
import type { ScriptedPart } from "./script/types";

export function appendPartToLastMessage(
  messages: UIMessage[],
  part: ScriptedPart["part"],
): UIMessage[] {
  const next = messages.slice();
  const last = next[next.length - 1];
  next[next.length - 1] = { ...last, parts: [...last.parts, part] };
  return next;
}
