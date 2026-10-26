import type { UIMessage } from "ai";

// The preamble names the user ("Priya's team") when a name is on file and says
// "this user's team" when it isn't; the matcher must hide both.
// Names are capped (80 characters at most), and bounding the match keeps a long
// crafted line from backtracking quadratically. The `u` flag makes the bound
// count characters (code points) like the backend's caps do, not UTF-16 units,
// so an emoji name within the cap still matches.
const NAME = "[^\\[\\]\\r\\n]{1,80}";
const TEAM_OWNER = NAME + "'s team";

const DELEGATION_PREAMBLE = new RegExp(
  "^\\[Delegated task from " +
    NAME +
    ", a teammate on " +
    TEAM_OWNER +
    " — not " +
    "the user\\. They cannot see your thread, so report the outcome in your " +
    "final message\\. If the task needs something only the user can " +
    "provide, say what is missing instead of guessing\\.\\]\\r?\\n\\r?\\n",
  "u",
);

const HANDOFF_PREAMBLE = new RegExp(
  "^\\[Task handed to you by " +
    NAME +
    ", a teammate on " +
    TEAM_OWNER +
    "\\. It " +
    "is yours now: they are not waiting on a report and cannot answer " +
    "follow-ups\\. Take it to completion and tell the user the outcome " +
    "yourself\\. If something only the user can provide is missing, ask " +
    "them\\.\\]\\r?\\n\\r?\\n",
  "u",
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
