import type { UIDataTypes, UIMessage, UITools } from "ai";
import { extractDbSequence } from "./helpers/convertChatSessionToUiMessages";

export type Decision = "approved" | "declined";

type DecisionVerb = "hire" | "create" | "update";

export interface CardDecision {
  verdict: Decision;
  verb: DecisionVerb | null;
  name: string | null;
  confirmationId: string | null;
}

/** The decision the chat holds for one card, and the row that holds it. */
export interface AnsweredCard {
  verdict: Decision;
  messageID: string;
}

interface DecisionLineArgs {
  kind: "hire" | "raise" | "update" | "soul" | null;
  name: string;
  confirmationId: string;
  decision: Decision;
}

const CARD_DECISION_KIND = "card_decision";

// The line gate/card_approval.py takes as the answer to the confirm, which the
// model reads too. __tests__/cardDecisionLines.json pins it to that file's
// _DECISION, so change both or neither.
export function decisionLine({
  kind,
  name,
  confirmationId,
  decision,
}: DecisionLineArgs): string {
  const verb = verbFor(kind);
  const shown = plainName(name);
  return decision === "approved"
    ? `Approved: ${verb} ${shown} (confirmation_id: ${confirmationId}).`
    : `Not approved: do not ${verb} ${shown}, discard that proposal (confirmation_id: ${confirmationId}).`;
}

// The line is a protocol the chat and the gate both parse: a line break,
// tab or control character in a name would read differently on each side.
function plainName(name: string): string {
  return name.replace(/[\s\p{Cc}]+/gu, " ").trim();
}

function verbFor(kind: DecisionLineArgs["kind"]): DecisionVerb {
  if (kind === "hire") return "hire";
  if (kind === "update" || kind === "soul") return "update";
  return "create";
}

const DECISION_RE =
  /^(Approved|Not approved): (.+) \(confirmation_id: ([^\s()]+)\)\.$/;
const APPROVED_SUBJECT_RE = /^(hire|create|update) (.+)$/;
const DECLINED_SUBJECT_RE =
  /^do not (hire|create|update) (.+), discard that proposal$/;

/** What a user row decided on proposal cards, or null when it says anything
 *  else. The server tags such a row; the text is read as well, for the bubble
 *  shown before the server's copy arrives and for rows saved before the tag.
 *  A row with an attachment stays the user's message, so the file shows. */
export function getCardDecisions(
  message: UIMessage<unknown, UIDataTypes, UITools>,
): CardDecision[] | null {
  if (message.role !== "user" || isServerWritten(message.metadata)) return null;
  if (message.parts.some((part) => part.type !== "text")) return null;
  return namedDecisions(
    decisionsFromMetadata(message.metadata) ??
      decisionsFromText(messageText(message)),
  );
}

/** What a message still waiting in the queue decided, read from its text the
 *  same way, or null when it says anything else. */
export function getTextDecisions(text: string): CardDecision[] | null {
  return namedDecisions(decisionsFromText(text));
}

// The card always names the change; a line in other words is shown as typed.
function namedDecisions(decisions: CardDecision[] | null) {
  if (!decisions?.every((decision) => decision.verb && decision.name))
    return null;
  return decisions;
}

/** Text made only of card decision lines: the card's protocol, which is never
 *  put back in the composer for the user to edit. */
export function isDecisionOnlyText(text: string): boolean {
  return decisionsFromText(text) !== null;
}

/** The row a card decision send just added, which the server never saved:
 *  a user row, unsaved, holding exactly that decision text. Any other row,
 *  saved or someone else's, is not the send's to take back. */
export function isUnsentDecisionRow(
  message: UIMessage<unknown, UIDataTypes, UITools> | undefined,
  sentText: string,
): boolean {
  return (
    message?.role === "user" &&
    extractDbSequence(message) === null &&
    isDecisionOnlyText(sentText) &&
    messageText(message) === sentText.trim()
  );
}

function decisionsFromMetadata(metadata: unknown): CardDecision[] | null {
  if (!metadata || typeof metadata !== "object") return null;
  const { kind, decisions } = metadata as Record<string, unknown>;
  if (kind !== CARD_DECISION_KIND || !Array.isArray(decisions)) return null;
  const parsed = decisions.flatMap(toCardDecision);
  return parsed.length ? parsed : null;
}

function toCardDecision(item: unknown): CardDecision[] {
  if (!item || typeof item !== "object") return [];
  const { verdict, verb, name, confirmation_id } = item as Record<
    string,
    unknown
  >;
  if (verdict !== "approved" && verdict !== "declined") return [];
  return [
    {
      verdict,
      verb:
        verb === "hire" || verb === "create" || verb === "update" ? verb : null,
      name: typeof name === "string" && name ? name : null,
      confirmationId:
        typeof confirmation_id === "string" && confirmation_id
          ? confirmation_id
          : null,
    },
  ];
}

function decisionsFromText(text: string): CardDecision[] | null {
  const lines = text.split("\n").filter((line) => trimLine(line));
  const parsed = lines.flatMap(parseLine);
  return parsed.length && parsed.length === lines.length ? parsed : null;
}

// Trimmed from both ends of a line, by the gate's _trim too: whitespace,
// control characters and a BOM. trim() and Python's strip() each miss some.
// Scanned from each end, not matched, so a long run inside costs one pass.
const EDGE_CHAR = /^[\s\p{Cc}\ufeff]$/u;

function trimLine(line: string): string {
  let start = 0;
  let end = line.length;
  while (start < end && EDGE_CHAR.test(line[start])) start++;
  while (end > start && EDGE_CHAR.test(line[end - 1])) end--;
  return line.slice(start, end);
}

function parseLine(line: string): CardDecision[] {
  const match = DECISION_RE.exec(trimLine(line));
  if (!match) return [];
  const verdict: Decision = match[1] === "Approved" ? "approved" : "declined";
  const subject = (
    verdict === "approved" ? APPROVED_SUBJECT_RE : DECLINED_SUBJECT_RE
  ).exec(match[2]);
  return [
    {
      verdict,
      verb: subject ? (subject[1] as DecisionVerb) : null,
      name: subject ? subject[2] : null,
      confirmationId: match[3],
    },
  ];
}

/** Every card the user's rows answered, by confirmation id. A card's buttons
 *  read this, not local state, so a reload, a reopened chat or another tab
 *  never asks again. Any decision line counts, as it does for the gate, and a
 *  later line overrides an earlier one, as a decline withdrew an approval. A
 *  saved answer wins over one only on screen: the server refuses a second
 *  answer, so an unsaved one that disagrees was never taken. A message still
 *  queued behind a running turn is an unsaved answer: the server took it, and
 *  the card does not ask again while it waits. */
export function getAnsweredCards(
  messages: UIMessage<unknown, UIDataTypes, UITools>[],
  queued: readonly string[] = [],
): Map<string, AnsweredCard> {
  const saved = new Map<string, AnsweredCard>();
  const unsaved = new Map<string, AnsweredCard>();
  function record(
    answered: Map<string, AnsweredCard>,
    decisions: CardDecision[],
    messageID: string,
  ) {
    for (const { verdict, confirmationId } of decisions) {
      if (confirmationId) answered.set(confirmationId, { verdict, messageID });
    }
  }
  for (const message of messages) {
    if (message.role !== "user") continue;
    record(
      extractDbSequence(message) === null ? unsaved : saved,
      rowDecisions(message.metadata, messageText(message)),
      message.id,
    );
  }
  queued.forEach((text, i) =>
    record(unsaved, rowDecisions(undefined, text), `queued-${i}`),
  );
  for (const [id, card] of unsaved) {
    if (!saved.has(id)) saved.set(id, card);
  }
  return saved;
}

/** An unsaved decision row whose every card a saved row answered: the server
 *  refused it, so it is not shown as an answer. */
export function isRefusedAnswer(
  messageID: string,
  decisions: CardDecision[],
  answered: ReadonlyMap<string, { messageID: string }>,
): boolean {
  return decisions.every(({ confirmationId }) => {
    const holder = confirmationId ? answered.get(confirmationId) : undefined;
    return holder !== undefined && holder.messageID !== messageID;
  });
}

function messageText(
  message: UIMessage<unknown, UIDataTypes, UITools>,
): string {
  return message.parts
    .map((part) => (part.type === "text" ? part.text : ""))
    .join("\n")
    .trim();
}

/** What a chat's stored rows answered, card by card (a later line wins),
 *  read from the raw rows of GET /sessions/{id}: what a tab opened before
 *  another one answered checks before it sends, and shows meanwhile. */
export function answersInRows(rows: unknown[]): Map<string, Decision> {
  const answers = new Map<string, Decision>();
  for (const row of rows) {
    if (!row || typeof row !== "object") continue;
    const { role, content, metadata } = row as Record<string, unknown>;
    if (role !== "user") continue;
    const text = typeof content === "string" ? content : "";
    for (const { verdict, confirmationId } of rowDecisions(metadata, text)) {
      if (confirmationId) answers.set(confirmationId, verdict);
    }
  }
  return answers;
}

// Every decision line in a row, not only a row made of them, as for the gate.
function rowDecisions(metadata: unknown, text: string): CardDecision[] {
  if (isServerWritten(metadata)) return [];
  return decisionsFromMetadata(metadata) ?? text.split("\n").flatMap(parseLine);
}

// A user-role row the server wrote for something else is not the user
// answering, whatever its text says: a delegated or subtask prompt, a held
// call's rows, a hidden control turn. A row queued behind a running turn
// carries the queue's own metadata and is still the user's. The backend's
// card_decided.is_server_written reads rows the same way.
const SERVER_WRITTEN_KEYS = [
  "from_session_id",
  "held_call",
  "held_calls_answered",
];

function isServerWritten(metadata: unknown): boolean {
  if (!metadata || typeof metadata !== "object") return false;
  const value = metadata as Record<string, unknown>;
  return (
    SERVER_WRITTEN_KEYS.some((key) => key in value) ||
    value.hidden === true ||
    (value.kind !== undefined && value.kind !== CARD_DECISION_KIND)
  );
}

const GERUNDS: Record<DecisionVerb, string> = {
  hire: "hiring",
  create: "creating",
  update: "updating",
};

const NAMES = new Intl.ListFormat("en", { type: "conjunction" });

interface NoteGroup {
  verdict: Decision;
  verb: DecisionVerb | null;
  names: string[];
}

/** One sentence per run of the same answer to the same kind of change:
 *  "You approved hiring Maya and Fiona", then "You declined creating Ada". */
export function decisionNotes(
  decisions: CardDecision[],
  { readOnly = false }: { readOnly?: boolean } = {},
) {
  const groups: NoteGroup[] = [];
  for (const { verdict, verb, name } of decisions) {
    const named = verb !== null && name !== null;
    const last = groups[groups.length - 1];
    if (named && last?.verdict === verdict && last.verb === verb) {
      last.names.push(name);
      continue;
    }
    groups.push(
      named
        ? { verdict, verb, names: [name] }
        : { verdict, verb: null, names: [] },
    );
  }
  return groups.map((group) => ({
    verdict: group.verdict,
    text: noteText(group, readOnly),
  }));
}

// A shared chat is read by someone else, so it does not say "you".
function noteText(
  { verdict, verb, names }: NoteGroup,
  readOnly: boolean,
): string {
  const answer = readOnly
    ? `${verdict[0].toUpperCase()}${verdict.slice(1)}`
    : `You ${verdict}`;
  return verb ? `${answer} ${GERUNDS[verb]} ${NAMES.format(names)}` : answer;
}
