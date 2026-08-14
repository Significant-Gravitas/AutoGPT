import { Text } from "@/components/atoms/Text/Text";
import { ExpertAvatar } from "@/components/molecules/ExpertAvatar/ExpertAvatar";
import { Icon } from "@/components/atoms/Icon/Icon";
import { FlashIcon } from "@hugeicons/core-free-icons";
import Link from "next/link";
import { getLastRunLabel } from "../../../helpers";
import {
  ExpertWorkflowGroupData,
  getWorkflowScheduleLabel,
  isWorkflowScheduled,
  workflowNeedsSetup,
} from "../helpers";

interface Props {
  group: ExpertWorkflowGroupData;
}

export function ExpertWorkflowGroup({ group }: Props) {
  const { expert, workflows } = group;
  const lastRun = getLastRunLabel(expert);
  const meta =
    lastRun ??
    `${workflows.length} ${workflows.length === 1 ? "workflow" : "workflows"}`;
  const isPaused = Boolean(expert.schedules_paused_at);

  return (
    <section
      aria-label={`${expert.name} runs`}
      className="rounded-2xl border border-zinc-200 bg-white"
    >
      <div className="flex items-center gap-3 border-b border-zinc-100 px-4 py-3">
        <ExpertAvatar
          name={expert.name}
          avatarUrl={expert.avatar_url ?? null}
          size={32}
        />
        <div className="min-w-0 flex-1">
          <Text variant="large-medium" className="truncate">
            {`${expert.name} · ${meta}`}
          </Text>
        </div>
        {isPaused ? (
          <span className="shrink-0 rounded-full bg-amber-50 px-2.5 py-1 text-xs text-amber-700 ring-1 ring-inset ring-amber-200">
            Paused
          </span>
        ) : null}
      </div>

      {workflows.length === 0 ? (
        <div className="px-4 py-3">
          <Text variant="small" className="text-zinc-500">
            Nothing installed yet —{" "}
            <Link href="/marketplace" className="underline">
              browse marketplace
            </Link>
          </Text>
        </div>
      ) : (
        <div className="divide-y divide-zinc-100">
          {workflows.map((workflow) => {
            const scheduleLabel = getWorkflowScheduleLabel(workflow);
            return (
              <div
                key={workflow.id}
                data-testid="what-runs-workflow-row"
                className="flex items-center gap-3 px-4 py-3"
              >
                <Icon
                  icon={FlashIcon}
                  size={18}
                  className="shrink-0 text-zinc-400"
                />
                <div className="min-w-0 flex-1">
                  <Text variant="body" className="truncate">
                    {workflow.name ?? "Unnamed workflow"}
                  </Text>
                </div>
                {workflowNeedsSetup(workflow) ? (
                  <span className="shrink-0 rounded-full bg-amber-50 px-2.5 py-1 text-xs text-amber-700 ring-1 ring-inset ring-amber-200">
                    Needs setup
                  </span>
                ) : isPaused && isWorkflowScheduled(workflow) ? (
                  <span className="shrink-0 rounded-full bg-zinc-100 px-2.5 py-1 text-xs text-zinc-500">
                    Paused
                  </span>
                ) : scheduleLabel ? (
                  <span className="shrink-0 rounded-full bg-zinc-100 px-2.5 py-1 text-xs text-zinc-600">
                    {scheduleLabel}
                  </span>
                ) : null}
              </div>
            );
          })}
        </div>
      )}
    </section>
  );
}
