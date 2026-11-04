import {
  AlertCircleIcon,
  BubbleChatIcon,
  Cancel01Icon,
  CancelCircleIcon,
  CheckListIcon,
  CheckmarkCircle02Icon,
  CircleIcon,
  DashedLineCircleIcon,
  EyeIcon,
  File01Icon,
  Link01Icon,
  Loading03Icon,
  MinusSignCircleIcon,
  PencilEdit02Icon,
  Tick02Icon,
} from "@hugeicons/core-free-icons";
import type { IconSvgElement } from "@hugeicons/react";
import type { Attachment } from "@/app/api/__generated__/models/attachment";
import type { Contract } from "@/app/api/__generated__/models/contract";
import type { Criterion } from "@/app/api/__generated__/models/criterion";
import type { CriterionStatus } from "@/app/api/__generated__/models/criterionStatus";
import type { EvidenceDetail } from "@/app/api/__generated__/models/evidenceDetail";
import type { Step } from "@/app/api/__generated__/models/step";
import type { StepStatus as ContractStepStatus } from "@/app/api/__generated__/models/stepStatus";
import type { Tone } from "@/components/styles/tones";
import type {
  AssigneeType,
  Task,
  ExpertOption,
  TaskAssignee,
  TaskState,
  TaskStatus,
} from "../../data/types";

export const STATUS_ICONS: Record<
  TaskStatus,
  { icon: IconSvgElement; tone: Tone }
> = {
  backlog: { icon: DashedLineCircleIcon, tone: "faint" },
  todo: { icon: CircleIcon, tone: "quiet" },
  in_progress: { icon: Loading03Icon, tone: "info" },
  blocked: { icon: AlertCircleIcon, tone: "waiting" },
  in_review: { icon: EyeIcon, tone: "info" },
  done: { icon: CheckmarkCircle02Icon, tone: "ok" },
  cancelled: { icon: CancelCircleIcon, tone: "faint" },
};

export interface AssigneeChoice {
  key: string;
  label: string;
  value: { type: AssigneeType; id: string | null };
}

export function getAssigneeChoices(experts: ExpertOption[] | undefined) {
  const base: AssigneeChoice[] = [
    { key: "otto", label: "Otto", value: { type: "otto", id: null } },
  ];
  const fromExperts = (experts ?? []).map<AssigneeChoice>((expert) => ({
    key: expert.id,
    label: expert.name,
    value: { type: "expert", id: expert.id },
  }));
  return [...base, ...fromExperts];
}

export function getAssigneeLabel(assignee: TaskAssignee | null) {
  if (!assignee) return "Unassigned";
  if (assignee.type === "user") return assignee.name || "Me";
  return assignee.name;
}

export function formatDueDate(iso: string) {
  return new Date(iso).toLocaleDateString(undefined, {
    month: "short",
    day: "numeric",
  });
}

function escapeRegExp(text: string) {
  return text.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

export interface MentionPart {
  text: string;
  mention: boolean;
}

export function splitMentions(body: string, names: string[]): MentionPart[] {
  const sorted = [...names].sort((a, b) => b.length - a.length);
  const alternatives = [...sorted.map(escapeRegExp), "[\\w-]+"];
  const pattern = new RegExp(`@(?:${alternatives.join("|")})`, "g");
  const parts: MentionPart[] = [];
  let last = 0;
  for (const match of body.matchAll(pattern)) {
    const start = match.index ?? 0;
    if (start > last)
      parts.push({ text: body.slice(last, start), mention: false });
    parts.push({ text: match[0], mention: true });
    last = start + match[0].length;
  }
  if (last < body.length)
    parts.push({ text: body.slice(last), mention: false });
  return parts;
}

export function formatElapsed(startedAt: string | null, now: number) {
  if (!startedAt) return "0s";
  const totalSeconds = Math.max(
    0,
    Math.floor((now - new Date(startedAt).getTime()) / 1000),
  );
  const h = Math.floor(totalSeconds / 3600);
  const m = Math.floor((totalSeconds % 3600) / 60);
  const s = totalSeconds % 60;
  if (h) return `${h}h ${m}m`;
  if (m) return `${m}m ${s}s`;
  return `${s}s`;
}

export function isTypingTarget(target: EventTarget | null) {
  if (!(target instanceof HTMLElement)) return false;
  return (
    target.isContentEditable ||
    target.tagName === "INPUT" ||
    target.tagName === "TEXTAREA" ||
    target.tagName === "SELECT"
  );
}

export function isTaskOverdue(dueDate: string | null, status: TaskStatus) {
  if (!dueDate || status === "done" || status === "cancelled") return false;
  return new Date(dueDate).getTime() < Date.now();
}

export type StepStatus = "completed" | "running" | "error" | null;

export const STEP_STATUS: Record<
  "completed" | "running" | "error",
  { icon: IconSvgElement; tone: Tone; label: string }
> = {
  completed: { icon: CheckmarkCircle02Icon, tone: "ok", label: "Done" },
  running: { icon: Loading03Icon, tone: "info", label: "Running" },
  error: { icon: AlertCircleIcon, tone: "error", label: "Failed" },
};

export function getInboxStateLabel(task: Task) {
  const { state, needs_you } = task.inbox;
  if (state === "running") return { label: "Running", tone: "info" as Tone };
  if (state === "done") return { label: "Done", tone: "ok" as Tone };
  if (state === "failed") return { label: "Failed", tone: "error" as Tone };
  return {
    label: needs_you ? "Needs you" : "Waiting",
    tone: (needs_you ? "waiting" : "quiet") as Tone,
  };
}

type Glyph = { icon: IconSvgElement; tone: Tone; label: string };

export const PLAN_STEP_GLYPH: Record<ContractStepStatus, Glyph> = {
  pending: { icon: CircleIcon, tone: "faint", label: "Not started" },
  running: { icon: Loading03Icon, tone: "info", label: "In progress" },
  done: { icon: CheckmarkCircle02Icon, tone: "ok", label: "Done" },
  failed: { icon: CancelCircleIcon, tone: "error", label: "Failed" },
  skipped: { icon: MinusSignCircleIcon, tone: "faint", label: "Skipped" },
};

export const STEP_TEXT_TONE: Record<ContractStepStatus, Tone> = {
  pending: "quiet",
  running: "ink",
  done: "text",
  failed: "error",
  skipped: "quiet",
};

export const ROW_TEXT_TONE: Record<ContractStepStatus, Tone> = {
  pending: "quiet",
  running: "text",
  done: "quiet",
  failed: "error",
  skipped: "faint",
};

export function countSettled(items: { status?: ContractStepStatus }[]) {
  return items.filter((i) => i.status === "done" || i.status === "skipped")
    .length;
}

export const CRITERION_GLYPH: Record<CriterionStatus, Glyph> = {
  pending: { icon: CircleIcon, tone: "faint", label: "Not checked yet" },
  met: { icon: Tick02Icon, tone: "ok", label: "Met" },
  unmet: { icon: Cancel01Icon, tone: "error", label: "Not met" },
  skipped: { icon: MinusSignCircleIcon, tone: "faint", label: "Skipped" },
};

const EVIDENCE_ICONS: Record<string, IconSvgElement> = {
  link: Link01Icon,
  url: Link01Icon,
  file: File01Icon,
  change: PencilEdit02Icon,
  chat: BubbleChatIcon,
};

export function evidenceIcon(kind: string) {
  return EVIDENCE_ICONS[kind] ?? CheckListIcon;
}

export function criteriaMeta(criteria: Criterion[]) {
  const met = criteria.filter(
    (c) => c.status === "met" || c.status === "skipped",
  ).length;
  return `${met} of ${criteria.length} met`;
}

export function planMeta(plan: Step[]) {
  return `${countSettled(plan)} of ${plan.length} done`;
}

/** Proof not tied to a check, shown under the checklist. */
export function looseEvidence(contract: Contract) {
  const linked = new Set(
    (contract.done_when ?? []).flatMap((c) => c.evidence_ids ?? []),
  );
  return (contract.evidence ?? []).filter(
    (e) => !linked.has(e.id) && e.id !== contract.result?.primary_evidence_id,
  );
}

export function evidenceById(contract: Contract) {
  return new Map((contract.evidence ?? []).map((e) => [e.id, e]));
}

export function evidenceContent(detail: EvidenceDetail | null | undefined) {
  const content = detail?.activity?.data?.content;
  return typeof content === "string" && content.trim() ? content : null;
}

export function attachmentHref(attachment: Attachment | null | undefined) {
  if (!attachment) return null;
  if (attachment.url) return attachment.url;
  if (attachment.kind === "chat" && attachment.ref_id)
    return `/copilot?sessionId=${attachment.ref_id}`;
  return null;
}

export function formatClock(iso: string) {
  return new Date(iso).toLocaleTimeString(undefined, {
    hour: "numeric",
    minute: "2-digit",
  });
}

export type PanelCard = "outcome" | "attention" | null;

interface PanelCardArgs {
  state: TaskState | undefined;
  hasResult: boolean;
  hasAsk: boolean;
  hasBlocker: boolean;
  awaitingAccept: boolean;
  approvals: number;
}

/** The one card under the ask: the result when done, the failure when it
 *  failed, the hire cards or the question when it needs you. */
export function panelCard({
  state,
  hasResult,
  hasAsk,
  hasBlocker,
  awaitingAccept,
  approvals,
}: PanelCardArgs): PanelCard {
  if (state === "stopped") return null;
  if (state === "done") return hasResult ? "outcome" : null;
  if (state === "stuck") return "attention";
  if (approvals > 0 || hasAsk || hasBlocker || awaitingAccept)
    return "attention";
  return null;
}

/** Every piece of proof, the main one first, each once. */
export function orderedEvidence(contract: Contract) {
  const all = contract.evidence ?? [];
  const primaryID = contract.result?.primary_evidence_id;
  const primary = all.find((e) => e.id === primaryID);
  return primary ? [primary, ...all.filter((e) => e !== primary)] : all;
}
