import type { SessionDetailResponse } from "@/app/api/__generated__/models/sessionDetailResponse";
import type { LiveStep } from "./components/ToolChain/SubSessionLive";
import { asObject, str } from "./components/ToolChain/resultHelpers";
import type { DelegationAnswer } from "./delegationAnswerStore";
import type {
  ChatDelegation,
  DelegationStatus,
  LiveDelegationStatus,
} from "./delegations";

export interface AskedQuestion {
  text: string | null;
  options: string[];
}

export function pendingQuestionOf(session: SessionDetailResponse) {
  const question = session.metadata?.pending_question;
  if (!question || typeof question.text !== "string" || !question.text)
    return null;
  const askedAt = Date.parse(String(question.asked_at ?? ""));
  return {
    text: question.text,
    askedAt: Number.isFinite(askedAt) ? askedAt : null,
  };
}

/** The teammate's last `ask_question` call in their current turn: the
 *  question and the option chips it offered. */
export function askedQuestionOf(steps: LiveStep[]): AskedQuestion | null {
  const step = steps.findLast((s) => s.name === "ask_question");
  if (!step) return null;
  const input = asObject(step.input);
  const questions = Array.isArray(input?.questions) ? input.questions : [];
  const items = questions.flatMap((q) => {
    const item = asObject(q);
    return item ? [item] : [];
  });
  const text =
    items
      .map((item) => str(item, "question"))
      .filter(Boolean)
      .join(" ") || null;
  const first = items[0];
  const options = Array.isArray(first?.options)
    ? first.options.filter((o): o is string => typeof o === "string")
    : [];
  return { text, options };
}

const IN_FLIGHT = new Set<DelegationStatus>(["running", "queued"]);
const POLL_TRUSTED = new Set<DelegationStatus>([
  "running",
  "queued",
  "needs-input",
  "completed",
]);

interface LiveInputs {
  delegation: ChatDelegation;
  session: SessionDetailResponse | null;
  isLive: boolean;
  question: string | null;
  isError: boolean;
  isPaused: boolean;
  answer: DelegationAnswer | null;
  now: number;
}

const ANSWER_GRACE_MS = 15_000;

/** The transcript freezes the status the call returned with; the teammate's
 *  own session is the truth while it can be read: running again (an answer,
 *  a resumed cap) is running, and a teammate that stopped on a question
 *  needs the user, whatever the transcript says. */
export function resolveLiveStatus({
  delegation,
  session,
  isLive,
  question,
  isError,
  isPaused,
  answer,
  now,
}: LiveInputs): LiveDelegationStatus {
  const frozen = delegation.status;
  const inFlight = IN_FLIGHT.has(frozen);
  if (inFlight && (isError || isPaused)) return "unknown";
  if (!session || !POLL_TRUSTED.has(frozen)) return frozen;
  // A re-delegation into the same thread opened a newer entry; whatever the
  // thread does now belongs to that one, so this run keeps its result.
  if (delegation.superseded && frozen === "completed") return frozen;
  if (isLive) {
    return session.chat_status?.toLowerCase() === "queued"
      ? "queued"
      : "running";
  }
  if (question)
    return answer?.question === question ? "running" : "needs-input";
  // Sent, but the teammate's turn has not picked it up yet.
  if (answer && now - answer.sentAt < ANSWER_GRACE_MS) return "running";
  return "completed";
}
