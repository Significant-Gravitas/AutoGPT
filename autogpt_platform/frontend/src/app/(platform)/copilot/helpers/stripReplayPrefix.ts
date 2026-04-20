import type { UIMessage } from "ai";

import { deduplicateMessages } from "../helpers";

/**
 * Strip replayed leading text from assistant messages.
 *
 * **Why this exists**: the Claude Agent SDK with `--resume` replays the
 * previous turn's assistant text at the start of the next turn's assistant
 * message. Left unchecked, the UI shows the same greeting/response twice
 * (once under Turn N−1's user bubble, again at the top of Turn N). We
 * detect and strip that replayed prefix so each turn stays anchored to its
 * own user message.
 *
 * **How** (three cases, evaluated against the longest matching earlier
 * assistant text):
 *
 * 1. earlier **equals** current → pure replay; drop the current message.
 * 2. earlier is a **strict prefix** of current → strip those leading chars
 *    from the current message's text parts (preserving non-text parts).
 * 3. current is a **strict prefix** of earlier (live-streaming catch-up) →
 *    drop the current message. Once its text grows past the earlier one,
 *    case 2 takes over.
 *
 * Comparison is by concatenated text content rather than by part index
 * because `--resume` interleaves different step-boundary part layouts, so
 * a structural prefix match is unreliable. Claude only replays TEXT at the
 * top of the next turn in practice, so text-only comparison covers the
 * cases we see.
 */
export function stripReplayPrefix(raw: UIMessage[]): UIMessage[] {
  const deduped = deduplicateMessages(raw);
  const texts = deduped.map(concatText);
  const out: UIMessage[] = [];

  for (let i = 0; i < deduped.length; i++) {
    const msg = deduped[i];
    if (msg.role !== "assistant" || !texts[i]) {
      out.push(msg);
      continue;
    }

    const match = findLongestReplay(texts, i);
    if (match.drop) continue;
    if (match.stripLen === 0) {
      out.push(msg);
      continue;
    }
    const trimmed = stripLeadingTextChars(msg, match.stripLen);
    if (trimmed !== null) out.push(trimmed);
  }
  return out;
}

function concatText(msg: UIMessage): string {
  return (msg.parts ?? [])
    .map((p) => ("text" in p && typeof p.text === "string" ? p.text : ""))
    .join("");
}

/**
 * Return either `{ drop: true }` if the current message should be removed,
 * or `{ stripLen }` with the number of leading text chars to strip (0 =
 * keep as-is).
 */
function findLongestReplay(
  texts: string[],
  i: number,
): { drop: boolean; stripLen: number } {
  const myText = texts[i];
  let stripLen = 0;

  for (let j = 0; j < i; j++) {
    const earlier = texts[j];
    if (!earlier) continue;

    if (myText.startsWith(earlier)) {
      // Case 1 + 2: earlier is a leading prefix of current.
      if (earlier.length < stripLen) continue;
      if (earlier.length === myText.length) return { drop: true, stripLen };
      stripLen = earlier.length;
    } else if (earlier.startsWith(myText)) {
      // Case 3: live-streaming replay is still catching up.
      return { drop: true, stripLen };
    }
  }
  return { drop: false, stripLen };
}

/**
 * Drop the leading `n` characters of text from *msg*'s parts, preserving
 * non-text parts (step-start/finish, tool-*) unchanged. Returns `null` if
 * the result would be empty (all text consumed and no structural parts).
 */
function stripLeadingTextChars(msg: UIMessage, n: number): UIMessage | null {
  const trimmed: UIMessage["parts"] = [];
  let remaining = n;

  for (const part of msg.parts ?? []) {
    if (remaining === 0) {
      trimmed.push(part);
      continue;
    }
    if ("text" in part && typeof part.text === "string") {
      if (part.text.length <= remaining) {
        remaining -= part.text.length;
        continue; // drop entire part
      }
      trimmed.push({ ...part, text: part.text.slice(remaining) });
      remaining = 0;
    } else {
      trimmed.push(part);
    }
  }

  const empty =
    trimmed.length === 0 ||
    trimmed.every(
      (p) => "text" in p && typeof p.text === "string" && p.text.length === 0,
    );
  return empty ? null : { ...msg, parts: trimmed };
}
