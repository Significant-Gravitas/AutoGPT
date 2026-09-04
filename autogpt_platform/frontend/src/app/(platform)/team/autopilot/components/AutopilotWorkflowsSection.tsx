import { Expert } from "@/app/api/__generated__/models/expert";
import { Badge } from "@/components/atoms/Badge/Badge";
import { Icon } from "@/components/atoms/Icon/Icon";
import { safeHumanizeCronExpression } from "@/lib/cron-expression-utils";
import { FlashIcon } from "@hugeicons/core-free-icons";
import Link from "next/link";
import { workflowNeedsSetup } from "../../helpers";

interface Props {
  experts: Expert[];
}

export function AutopilotWorkflowsSection({ experts }: Props) {
  const owners = experts.filter((expert) => expert.workflows.length > 0);

  if (owners.length === 0) {
    return (
      <p className="text-sm text-zinc-500">
        No workflows yet. Install one on an expert to give Autopilot something
        to run.
      </p>
    );
  }

  return (
    <div className="space-y-5">
      {owners.map((expert) => (
        <section key={expert.id} aria-label={`${expert.name} workflows`}>
          <Link
            href={`/team/${expert.id}`}
            className="mb-2 inline-flex items-baseline gap-2 text-sm hover:underline"
          >
            <span className="font-medium text-zinc-900">{expert.name}</span>
            <span className="text-zinc-500">{expert.role}</span>
          </Link>
          <div className="divide-y divide-zinc-100 rounded-lg border border-zinc-200/80 bg-white">
            {expert.workflows.map((workflow) => (
              <div
                key={workflow.id}
                className="flex items-center gap-3 px-4 py-3"
                data-testid="autopilot-workflow-row"
              >
                <Icon
                  icon={FlashIcon}
                  size={18}
                  className="shrink-0 text-zinc-500"
                />
                <div className="min-w-0 flex-1">
                  <div className="text-[15px] font-medium text-zinc-800">
                    {workflow.name ?? "Unnamed workflow"}
                  </div>
                  {workflow.description ? (
                    <div className="line-clamp-1 text-[13px] text-zinc-500">
                      {workflow.description}
                    </div>
                  ) : null}
                </div>
                {workflowNeedsSetup(workflow) ? (
                  <Badge variant="warning" size="small" className="shrink-0">
                    Needs setup
                  </Badge>
                ) : workflow.schedule_cron ? (
                  <Badge variant="info" size="small" className="shrink-0">
                    {safeHumanizeCronExpression(workflow.schedule_cron)}
                  </Badge>
                ) : null}
              </div>
            ))}
          </div>
        </section>
      ))}
    </div>
  );
}
