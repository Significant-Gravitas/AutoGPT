import type { TourMessage, TourPart } from "./script/types";

export function appendPartToLastMessage(
  messages: TourMessage[],
  part: TourPart,
): TourMessage[] {
  const next = messages.slice();
  const last = next[next.length - 1];
  next[next.length - 1] = { ...last, parts: [...last.parts, part] };
  return next;
}
