import type { UIMessage } from "ai";

export interface TurnItem {
  message: UIMessage;
  /** Text of the next user message, or null while none follows. */
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
  // Only a person answers a card; a routine's notice in between does not.
  const replies: (string | null)[] = [];
  let reply: string | null = null;
  for (let index = messages.length - 1; index >= 0; index--) {
    replies[index] = reply;
    if (messages[index].role === "user") reply = joinedText(messages[index]);
  }
  messages.forEach((message, index) => {
    const item = { message, replyText: replies[index] };
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
