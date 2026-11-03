import {
  AlertCircleIcon,
  CancelCircleIcon,
  CheckmarkCircle02Icon,
  CircleIcon,
  DashedLineCircleIcon,
  EyeIcon,
  Loading03Icon,
} from "@hugeicons/core-free-icons";
import type { IconSvgElement } from "@hugeicons/react";
import type { Tone } from "@/components/styles/tones";
import type {
  AssigneeType,
  Task,
  ExpertOption,
  TaskAssignee,
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
    { key: "me", label: "Me", value: { type: "user", id: "me" } },
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

export function formatClock(iso: string) {
  return new Date(iso).toLocaleTimeString(undefined, {
    hour: "numeric",
    minute: "2-digit",
  });
}
