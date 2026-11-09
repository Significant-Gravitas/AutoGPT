import type { UIMessage } from "ai";

const DELEGATION_PREAMBLE = new RegExp(
  "^\\[Delegated task from [^\\[\\]\\r\\n]+, a teammate on this user's team — not " +
    "the user\\. They cannot see your thread, so report the outcome in your " +
    "final message\\. If the task needs something only the user can " +
    "provide, say what is missing instead of guessing\\.\\]\\r?\\n\\r?\\n",
);

const HANDOFF_PREAMBLE = new RegExp(
  "^\\[Task handed to you by [^\\[\\]\\r\\n]+, a teammate on this user's team\\. It " +
    "is yours now: they are not waiting on a report and cannot answer " +
    "follow-ups\\. Take it to completion and tell the user the outcome " +
    "yourself\\. If something only the user can provide is missing, ask " +
    "them\\.\\]\\r?\\n\\r?\\n",
);

// Follows the delegation preamble on a new hire's first task.
const FIRST_TASK_NOTE = new RegExp(
  "^\\[This is your first task since joining the team, and the user reads " +
    "this thread: open with one line introducing yourself to them, then work " +
    "on it or ask what you need\\. Don't run a workflow, create a schedule " +
    "or send anything on this first turn; ask the user first\\.\\]" +
    "\\r?\\n\\r?\\n",
);

export function getVisibleUserMessageParts(parts: UIMessage["parts"]) {
  const firstTextIndex = parts.findIndex((part) => part.type === "text");
  const firstText = parts[firstTextIndex];
  if (firstText?.type !== "text") return parts;

  const preamble =
    firstText.text.match(DELEGATION_PREAMBLE) ??
    firstText.text.match(HANDOFF_PREAMBLE);
  if (!preamble) return parts;

  const afterPreamble = firstText.text.slice(preamble[0].length);
  const note = afterPreamble.match(FIRST_TASK_NOTE);
  const text = note ? afterPreamble.slice(note[0].length) : afterPreamble;
  if (!text.trim()) return parts;

  return parts.map((part, index) =>
    index === firstTextIndex ? { ...firstText, text } : part,
  );
}
