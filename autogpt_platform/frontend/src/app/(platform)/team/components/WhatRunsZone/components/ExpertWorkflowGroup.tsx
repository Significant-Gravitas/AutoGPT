import { Badge } from "@/components/atoms/Badge/Badge";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { ExpertAvatar } from "@/components/molecules/ExpertAvatar/ExpertAvatar";
import { safeHumanizeCronExpression } from "@/lib/cron-expression-utils";
import { FlashIcon } from "@hugeicons/core-free-icons";
import Link from "next/link";
import { getLastRunLabel, workflowNeedsSetup } from "../../../helpers";
import { ExpertWorkflowGroupData } from "../helpers";

interface Props {
  group: ExpertWorkflowGroupData;
}

export function ExpertWorkflowGroup({ group }: Props) {
  const { expert, workflows } = group;
  const lastRun = getLastRunLabel(expert);
  const countLabel = `${workflows.length} ${workflows.length === 1 ? "workflow" : "workflows"}`;
  const isPaused = Boolean(expert.schedules_paused_at);
  const lastRunVariant =
    expert.last_run_status === "FAILED"
      ? "error"
      : expert.last_run_status === "COMPLETED"
        ? "success"
        : "info";

  return (
    <section
      aria-label={`${expert.name} runs`}
      className="rounded-2xl border border-zinc-200 bg-white"
    >
      <div className="flex flex-wrap items-center gap-3 border-b border-zinc-100 px-4 py-3">
        <ExpertAvatar
          name={expert.name}
          avatarUrl={expert.avatar_url ?? null}
          size={32}
        />
        <div className="min-w-0 flex-1">
          <Text variant="large-medium" className="truncate">
            {`${expert.name} · ${countLabel}`}
          </Text>
        </div>
        {lastRun ? (
          <Badge
            variant={lastRunVariant}
            className="shrink-0 normal-case tracking-normal"
          >
            {lastRun}
          </Badge>
        ) : null}
        {isPaused ? (
          <Badge
            variant="info"
            className="shrink-0 normal-case tracking-normal"
          >
            Paused
          </Badge>
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
          {workflows.map(({ workflow, schedules }) => {
            const needsSetup = workflowNeedsSetup(workflow, schedules);
            const setupHref = workflow.library_agent_id
              ? `/library/agents/${workflow.library_agent_id}`
              : "/marketplace";
            return (
              <div
                key={workflow.id}
                data-testid="what-runs-workflow-row"
                className="flex flex-wrap items-center gap-3 px-4 py-3"
              >
                <Icon
                  icon={FlashIcon}
                  size={18}
                  aria-hidden="true"
                  className="shrink-0 text-zinc-400"
                />
                <div className="min-w-0 flex-1">
                  <Text variant="body" className="truncate">
                    {workflow.name ?? "Unnamed workflow"}
                  </Text>
                </div>
                <div className="ml-auto flex flex-wrap items-center justify-end gap-2">
                  {needsSetup ? (
                    <>
                      <Badge
                        variant="info"
                        className="normal-case tracking-normal"
                      >
                        Needs setup
                      </Badge>
                      <Link
                        href={setupHref}
                        className="rounded-sm text-sm font-medium text-zinc-700 underline underline-offset-2 focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-zinc-900"
                      >
                        Set up
                      </Link>
                    </>
                  ) : null}
                  {isPaused && schedules.length > 0 ? (
                    <Badge
                      variant="info"
                      className="normal-case tracking-normal"
                    >
                      Paused
                    </Badge>
                  ) : (
                    schedules.map((schedule) => (
                      <Badge
                        key={schedule.id}
                        variant="info"
                        className="normal-case tracking-normal"
                      >
                        {safeHumanizeCronExpression(schedule.cron)}
                      </Badge>
                    ))
                  )}
                </div>
              </div>
            );
          })}
        </div>
      )}
    </section>
  );
}
