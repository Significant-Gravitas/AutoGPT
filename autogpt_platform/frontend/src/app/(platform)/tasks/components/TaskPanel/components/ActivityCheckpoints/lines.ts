import type { TaskActivity } from "../../../../data/types";

/** Who did it, as a person says it: "You" for the user. */
export function whoOf(actor: TaskActivity["actor"]) {
  if (actor?.type === "user") return "You";
  return actor?.name || "Otto";
}

/** The first line of a message, without the markdown that starts a block
 *  (heading, quote, bullet), so it reads as one sentence in the feed. */
export function firstLine(text: string) {
  const line =
    text
      .split("\n")
      .map((l) => l.trim())
      .find(Boolean) ?? "";
  return line.replace(/^(#{1,6}\s+|>\s*|[-*+]\s+|\d+[.)]\s+)/, "").trim();
}

/** Task keys like "T-12" mean nothing to the reader. */
export function stripTaskKeys(text: string) {
  return text
    .replace(/\s*\(\s*T-\d+\s*\)/g, "")
    .replace(/\bT-\d+:?\s*/g, "")
    .replace(/\s{2,}/g, " ")
    .trim();
}

const PAST_VERBS = new Set([
  "built",
  "chose",
  "found",
  "made",
  "picked",
  "ran",
  "read",
  "sent",
  "set",
  "wrote",
  "put",
  "got",
]);

/** A worker's own summary as a sentence with an actor: "I drafted the copy"
 *  and "Drafted the copy" both read "Maya drafted the copy". */
export function sentence(who: string, text: string) {
  if (!text) return `${who} worked on it`;
  if (text.toLowerCase().startsWith(who.toLowerCase())) return text;
  const own = text.match(/^I(?:'ve| have)\s+(.*)$/s);
  if (own) return `${who} has ${own[1]}`;
  const mine = text.match(/^I(?:'m| am)\s+(.*)$/s);
  if (mine) return `${who} is ${mine[1]}`;
  const plain = text.match(/^I\s+(.*)$/s);
  if (plain) return `${who} ${plain[1]}`;
  const word = text.split(/\s/)[0].replace(/\W+$/, "").toLowerCase();
  if (word.endsWith("ed") || PAST_VERBS.has(word))
    return `${who} ${text[0].toLowerCase()}${text.slice(1)}`;
  return `${who}: ${text}`;
}
