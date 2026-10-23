import type { UIDataTypes, UIMessage, UITools } from "ai";

export type Decision = "approved" | "declined";

type DecisionVerb = "hire" | "create" | "update";

export interface CardDecision {
  verdict: Decision;
  verb: DecisionVerb | null;
  name: string | null;
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
  return decision === "approved"
    ? `Approved: ${verb} ${name} (confirmation_id: ${confirmationId}).`
    : `Not approved: do not ${verb} ${name}, discard that proposal (confirmation_id: ${confirmationId}).`;
}

function verbFor(kind: DecisionLineArgs["kind"]): DecisionVerb {
  if (kind === "hire") return "hire";
  if (kind === "update" || kind === "soul") return "update";
  return "create";
}

const DECISION_RE =
  /^(Approved|Not approved): (.+) \(confirmation_id: [^\s()]+\)\.$/;
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
  if (message.role !== "user") return null;
  if (message.parts.some((part) => part.type !== "text")) return null;
  return (
    decisionsFromMetadata(message.metadata) ??
    decisionsFromText(messageText(message))
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
  const { verdict, verb, name } = item as Record<string, unknown>;
  if (verdict !== "approved" && verdict !== "declined") return [];
  return [
    {
      verdict,
      verb:
        verb === "hire" || verb === "create" || verb === "update" ? verb : null,
      name: typeof name === "string" && name ? name : null,
    },
  ];
}

function decisionsFromText(text: string): CardDecision[] | null {
  const lines = text.split("\n").filter((line) => line.trim());
  const parsed = lines.flatMap(parseLine);
  return parsed.length && parsed.length === lines.length ? parsed : null;
}

function parseLine(line: string): CardDecision[] {
  const match = DECISION_RE.exec(line.trim());
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
    },
  ];
}

function messageText(
  message: UIMessage<unknown, UIDataTypes, UITools>,
): string {
  return message.parts
    .map((part) => (part.type === "text" ? part.text : ""))
    .join("\n")
    .trim();
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
export function decisionNotes(decisions: CardDecision[]) {
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
    text: noteText(group),
  }));
}

function noteText({ verdict, verb, names }: NoteGroup): string {
  if (!verb || !names.length) return `You ${verdict} a proposed change`;
  return `You ${verdict} ${GERUNDS[verb]} ${NAMES.format(names)}`;
}
