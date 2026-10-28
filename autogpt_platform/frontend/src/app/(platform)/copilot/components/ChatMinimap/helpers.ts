import type { UIDataTypes, UIMessage, UITools } from "ai";
import { isHeldCallRow } from "../ChatMessagesContainer/heldCallRows";

export interface MinimapEntry {
  id: string;
  title: string;
  body: string;
}

const TITLE_CHARS = 60;
const BODY_CHARS = 220;

/** One tick per user message. The rail is a way back to what *you* asked —
 *  ticking every assistant turn as well doubled the marks without adding a
 *  distinct destination, since a reply sits directly under its question. */
export function toMinimapEntries(
  messages: UIMessage<unknown, UIDataTypes, UITools>[],
): MinimapEntry[] {
  return messages.filter(isMinimapRow).map((message) => {
    const [title, body] = splitPreview(messageText(message));
    return {
      id: message.id,
      title: title || "Your message",
      body,
    };
  });
}

/** True when both lists hold the same ticks: the same user messages, by id
 *  and text, in the same order. A streamed flush only grows the reply, so the
 *  rail need not render for it. */
export function haveSameMinimapRows(
  a: UIMessage<unknown, UIDataTypes, UITools>[],
  b: UIMessage<unknown, UIDataTypes, UITools>[],
): boolean {
  if (a === b) return true;
  const left = a.filter(isMinimapRow);
  const right = b.filter(isMinimapRow);
  if (left.length !== right.length) return false;
  return left.every((message, i) => {
    const other = right[i];
    return (
      message === other ||
      (message.id === other.id && messageText(message) === messageText(other))
    );
  });
}

function isMinimapRow(
  message: UIMessage<unknown, UIDataTypes, UITools>,
): boolean {
  return message.role === "user" && !isHeldCallRow(message);
}

function messageText(
  message: UIMessage<unknown, UIDataTypes, UITools>,
): string {
  return message.parts
    .filter((part): part is Extract<typeof part, { type: "text" }> => {
      return part.type === "text";
    })
    .map((part) => part.text)
    .join("\n")
    .trim();
}

/** First line becomes the card's heading, the rest its greyed body. */
function splitPreview(text: string): [string, string] {
  const lines = text.split("\n").filter((line) => line.trim());
  const title = truncate(lines[0] ?? "", TITLE_CHARS);
  const body = truncate(lines.slice(1).join(" ").trim(), BODY_CHARS);
  return [title, body];
}

function truncate(value: string, max: number): string {
  const clean = value.replace(/\s+/g, " ").trim();
  return clean.length > max ? `${clean.slice(0, max).trimEnd()}…` : clean;
}

const RESTING_SCALE = 0.6;
const FALLOFF_PER_STEP = 0.2;

/** Ticks rest shrunk and swell toward the cursor, tapering one step at a
 *  time, so the rail reads as a soft bump rather than a hard switch. */
export function tickScale(index: number, hovered: number | null): number {
  if (hovered === null) return RESTING_SCALE;
  return Math.max(
    1 - FALLOFF_PER_STEP * Math.abs(index - hovered),
    RESTING_SCALE,
  );
}

export function tickColor(distance: number | null): string {
  if (distance === 0) return "bg-action";
  // TODO(tokens): no fitting role for bg-zinc-500 #83838C; #83838C is foreground-tertiary (a text colour) and placeholder (a text colour)
  if (distance === 1) return "bg-zinc-500";
  // TODO(tokens): no fitting role for bg-zinc-400 #ADADB3; #ADADB3 is foreground-subtle (a text colour)
  return "bg-zinc-400";
}
