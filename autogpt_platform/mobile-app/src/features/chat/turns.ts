import type { UIMessage } from "ai";

export interface TurnItem {
  message: UIMessage;
  /** Text of the next message, or null when this one is the last. */
  replyText: string | null;
}

export interface Turn {
  key: string;
  items: TurnItem[];
}

/**
 * Groups the thread into turns: a user message and the replies after it. The
 * list scrolls a whole turn, so a new question can sit at the top of the
 * screen with its reply growing below.
 */
export function groupTurns(messages: UIMessage[]): Turn[] {
  const turns: Turn[] = [];
  messages.forEach((message, index) => {
    const next = messages[index + 1];
    const item = { message, replyText: next ? joinedText(next) : null };
    const current = turns[turns.length - 1];
    if (message.role === "user" || !current) {
      turns.push({ key: message.id, items: [item] });
    } else {
      current.items.push(item);
    }
  });
  return turns;
}

function joinedText(message: UIMessage) {
  return message.parts.map((part) => (part.type === "text" ? part.text : "")).join("");
}
