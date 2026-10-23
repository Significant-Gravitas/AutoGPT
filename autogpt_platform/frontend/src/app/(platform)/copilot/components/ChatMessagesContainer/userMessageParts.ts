import type { UIMessage } from "ai";

// New rows name the user ("Priya's team"); rows stored before that, and users
// with no name on file, say "this user's team".
const TEAM_OWNER = "[^\\[\\]\\r\\n]+'s team";

const DELEGATION_PREAMBLE = new RegExp(
  "^\\[Delegated task from [^\\[\\]\\r\\n]+, a teammate on " +
    TEAM_OWNER +
    " — not " +
    "the user\\. They cannot see your thread, so report the outcome in your " +
    "final message\\. If the task needs something only the user can " +
    "provide, say what is missing instead of guessing\\.\\]\\r?\\n\\r?\\n",
);

const HANDOFF_PREAMBLE = new RegExp(
  "^\\[Task handed to you by [^\\[\\]\\r\\n]+, a teammate on " +
    TEAM_OWNER +
    "\\. It " +
    "is yours now: they are not waiting on a report and cannot answer " +
    "follow-ups\\. Take it to completion and tell the user the outcome " +
    "yourself\\. If something only the user can provide is missing, ask " +
    "them\\.\\]\\r?\\n\\r?\\n",
);

export function getVisibleUserMessageParts(parts: UIMessage["parts"]) {
  const firstTextIndex = parts.findIndex((part) => part.type === "text");
  const firstText = parts[firstTextIndex];
  if (firstText?.type !== "text") return parts;

  const preamble =
    firstText.text.match(DELEGATION_PREAMBLE) ??
    firstText.text.match(HANDOFF_PREAMBLE);
  if (!preamble) return parts;

  const text = firstText.text.slice(preamble[0].length);
  if (!text.trim()) return parts;

  return parts.map((part, index) =>
    index === firstTextIndex ? { ...firstText, text } : part,
  );
}
