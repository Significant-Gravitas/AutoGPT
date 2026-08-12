"use client";

import { Expert } from "@/app/api/__generated__/models/expert";
import { Button } from "@/components/atoms/Button/Button";
import { humanizeCronExpression } from "@/lib/cron-expression-utils";
import { FlashIcon, PlusSignIcon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";
import { workflowNeedsSetup } from "../../helpers";

interface Props {
  expert: Expert;
  accentIconClass: string;
  onInstallWorkflow: () => void;
}

export function ExpertWorkflowsSection({
  expert,
  accentIconClass,
  onInstallWorkflow,
}: Props) {
  return (
    <section>
      <div className="mb-2.5 flex items-center justify-between">
        <div className="text-xs font-medium uppercase tracking-[0.14em] text-zinc-400">
          Workflows
        </div>
        <Button
          variant="ghost"
          size="small"
          leftIcon={<Icon icon={PlusSignIcon} size={16} />}
          onClick={onInstallWorkflow}
        >
          Install workflow
        </Button>
      </div>
      {expert.workflows.length === 0 ? (
        <p className="text-sm text-zinc-500">
          No workflows yet. Install one to give {expert.name} something to run.
        </p>
      ) : (
        <div className="divide-y divide-zinc-100 rounded-xl border border-zinc-200/80 bg-white">
          {expert.workflows.map((workflow) => (
            <div
              key={workflow.id}
              className="flex items-center gap-3 px-4 py-3"
              data-testid="expert-workflow-row"
            >
              <Icon
                icon={FlashIcon}
                size={18}
                className={`shrink-0 ${accentIconClass}`}
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
                <span className="shrink-0 rounded-full bg-amber-50 px-2.5 py-1 text-xs text-amber-700 ring-1 ring-inset ring-amber-200">
                  Needs setup
                </span>
              ) : workflow.schedule_cron ? (
                <span className="shrink-0 rounded-full bg-zinc-100 px-2.5 py-1 text-xs text-zinc-600">
                  {humanizeCronExpression(workflow.schedule_cron)}
                </span>
              ) : null}
            </div>
          ))}
        </div>
      )}
    </section>
  );
}
