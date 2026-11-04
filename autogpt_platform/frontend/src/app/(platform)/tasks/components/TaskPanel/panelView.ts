import type { TaskSummary } from "@/app/api/__generated__/models/taskSummary";
import { statusOf } from "../../data/helpers";
import type { TaskDetail, TaskState } from "../../data/types";
import { unmetChecks } from "./components/AttentionCard/helpers";
import { orderedEvidence, panelCard } from "./helpers";

/** What the panel shows for a task, read from its detail and its summary. */
export function panelView(
  task: TaskDetail | undefined,
  summary: TaskSummary | undefined,
) {
  if (!task) return null;
  const contract = summary?.contract ?? null;
  const state: TaskState = summary?.state ?? task.state;
  const settled = state === "done" || state === "stopped";
  const ask = settled ? null : (contract?.ask ?? task.ask ?? null);
  const blocker = settled ? null : (contract?.blocker ?? task.blocker ?? null);
  const approvals = settled
    ? []
    : (contract?.approvals ?? task.approvals ?? []);
  const awaitingAccept =
    !settled && (summary?.raw_status ?? task.raw_status) === "awaiting_accept";
  const resultSummary =
    contract?.result?.summary || task.outcome_summary || null;
  const evidence = contract ? orderedEvidence(contract) : [];
  const card = panelCard({
    state,
    hasResult: !!resultSummary || evidence.length > 0,
    hasAsk: !!ask,
    hasBlocker: !!blocker,
    awaitingAccept,
    approvals: approvals.length,
  });
  const workSession = summary?.work_session_id ?? task.thread_session_id;

  return {
    contract,
    state,
    now: summary?.now ?? task.now,
    running: summary?.live?.running ?? state === "working",
    failed: statusOf(state).failed,
    ask,
    blocker,
    approvals,
    awaitingAccept,
    resultSummary,
    evidence,
    unmet: unmetChecks(contract?.done_when),
    card,
    proofInCard:
      card === "outcome" ||
      (card === "attention" && awaitingAccept && !ask && !blocker),
    workChatHref: workSession ? `/copilot?sessionId=${workSession}` : null,
  };
}
