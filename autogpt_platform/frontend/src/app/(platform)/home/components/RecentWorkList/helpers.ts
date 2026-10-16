import type { ExpertIdentityMap } from "@/app/(platform)/copilot/useExpertMap";
import { isInternalToolOutput } from "@/app/(platform)/copilot/components/ContextPanel/components/FilesTab/helpers";
import type { HomeRecentWork } from "@/app/api/__generated__/models/homeRecentWork";
import type { HomeRecentWorkItem } from "@/app/api/__generated__/models/homeRecentWorkItem";
import type { HomeWorkActor } from "@/app/api/__generated__/models/homeWorkActor";
import type { WorkspaceFileItem } from "@/app/api/__generated__/models/workspaceFileItem";
import { AUTOPILOT_NAME } from "@/components/molecules/AutopilotAvatar/helpers";
import { formatProviderName } from "@/components/contextual/IntegrationsPanel/helpers";
import { isHiddenWorkspaceFile } from "@/lib/workspace-files";

export const RECENT_WORK_LIMIT = 5;

export interface RecentWorkLine {
  id: string;
  actor: HomeWorkActor;
  verb: string;
  object: string | null;
  // "4 times" when the same thing happened in a row.
  suffix: string | null;
  at: string;
  link: string | null;
  failed: boolean;
}

const FILE_VERBS: Record<string, string> = {
  "file.deleted": "deleted",
};

const SCHEDULE_VERBS: Record<string, string> = {
  "schedule.created": "scheduled",
  "schedule.deleted": "cancelled",
  "schedule.paused": "paused",
  "schedule.resumed": "resumed",
};

function describeItem(
  item: HomeRecentWorkItem,
): Pick<RecentWorkLine, "verb" | "object"> | null {
  if (item.category === "file") {
    // Made and updated files come from the file list, which also has the
    // ones the activity log never recorded; only deletions come from here.
    if (item.event_type !== "file.deleted") return null;
    return { verb: FILE_VERBS[item.event_type], object: item.title };
  }
  if (item.category === "schedule") {
    const verb = SCHEDULE_VERBS[item.event_type];
    return verb ? { verb, object: item.title } : null;
  }
  const service = formatProviderName(item.provider);
  return service ? { verb: `used ${service}`, object: null } : null;
}

function toTime(value: string | Date | null | undefined) {
  if (!value) return null;
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? null : date.toISOString();
}

const SESSION_PATH = /^\/sessions\/([^/]+)\//;

function fileActor(
  expertId: string | null | undefined,
  expertsById: ExpertIdentityMap,
): HomeWorkActor {
  const expert = expertId ? expertsById.get(expertId) : undefined;
  if (!expert) return { kind: "autopilot", name: AUTOPILOT_NAME };
  return {
    kind: "expert",
    name: expert.name,
    expert: {
      id: expert.id,
      name: expert.name,
      role: expert.role ?? "",
      avatar_url: expert.avatarUrl,
    },
  };
}

function fileLines(
  files: WorkspaceFileItem[],
  expertsById: ExpertIdentityMap,
): RecentWorkLine[] {
  const lines: RecentWorkLine[] = [];
  for (const file of files) {
    if (isHiddenWorkspaceFile(file) || isInternalToolOutput(file)) continue;
    // Files outside a chat came from workflow runs, which the run lines cover.
    const sessionId = SESSION_PATH.exec(file.path)?.[1];
    const at = toTime(file.created_at);
    if (!sessionId || !at) continue;
    lines.push({
      id: `file:${file.id}`,
      actor: fileActor(file.expert_id, expertsById),
      verb: "made",
      object: file.name,
      suffix: null,
      at,
      link: `/home?sessionId=${encodeURIComponent(sessionId)}`,
      failed: false,
    });
  }
  return lines;
}

export function buildRecentWorkLines(
  recentWork: HomeRecentWork | undefined,
  files: WorkspaceFileItem[],
  expertsById: ExpertIdentityMap,
): RecentWorkLine[] {
  const lines = fileLines(files, expertsById);
  for (const group of recentWork?.groups ?? []) {
    const { actor } = group;
    for (const run of group.runs ?? []) {
      const at = toTime(run.occurred_at);
      if (!at) continue;
      const failed = run.status === "failed";
      // A workflow running on its own is both the subject and the thing run.
      const ranItself = actor.kind === "workflow";
      lines.push({
        id: `run:${run.id}`,
        actor,
        verb: ranItself
          ? failed
            ? "couldn't finish"
            : "ran"
          : failed
            ? "couldn't finish"
            : "finished",
        object: ranItself ? null : run.agent_name || run.title,
        suffix: null,
        at,
        link: run.link ?? null,
        failed,
      });
    }
    for (const item of group.items ?? []) {
      const at = toTime(item.occurred_at);
      const described = describeItem(item);
      if (!at || !described) continue;
      lines.push({
        id: `item:${item.id}`,
        actor,
        ...described,
        suffix: null,
        at,
        link: item.link ?? null,
        failed: false,
      });
    }
  }
  lines.sort((a, b) => b.at.localeCompare(a.at));
  return collapseRepeats(lines).slice(0, RECENT_WORK_LIMIT);
}

// "Remy used Google" five times in a row reads as one line with a count.
function collapseRepeats(lines: RecentWorkLine[]) {
  const collapsed: (RecentWorkLine & { count: number })[] = [];
  for (const line of lines) {
    const previous = collapsed[collapsed.length - 1];
    if (
      previous &&
      previous.actor.name === line.actor.name &&
      previous.actor.kind === line.actor.kind &&
      previous.verb === line.verb &&
      previous.object === line.object
    ) {
      previous.count += 1;
      continue;
    }
    collapsed.push({ ...line, count: 1 });
  }
  return collapsed.map(({ count, ...line }) => ({
    ...line,
    suffix: count > 1 ? `${count} times` : null,
  }));
}
