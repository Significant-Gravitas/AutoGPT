import { Expert } from "@/app/api/__generated__/models/expert";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Calendar03Icon, FlashIcon } from "@hugeicons/core-free-icons";
import { getCadenceLabel } from "./ExpertWorkflowCard/helpers";
import { ExpertSection } from "./ExpertSection";

interface Props {
  name: string;
  workflows: Expert["workflows"];
}

/** A quiet list at the foot of the page: the name, one line on what it does
 *  and its cadence. The story above sells the expert; this is the detail. */
export function ExpertWorkflowList({ name, workflows }: Props) {
  if (workflows.length === 0) return null;

  return (
    <ExpertSection
      title="Workflows"
      count={workflows.length}
      description={`Installed and ready to run the moment ${name} joins your team.`}
      quiet
    >
      <ul className="divide-y divide-border-base/70 rounded-2xl bg-surface px-4 smooth-shadow-ring-sm">
        {workflows.map((workflow) => (
          <li key={workflow.id} className="flex items-start gap-3 py-3">
            <Icon
              icon={FlashIcon}
              size={16}
              aria-hidden
              className="mt-0.5 shrink-0 text-foreground-subtle"
            />
            <div className="flex min-w-0 flex-1 flex-col gap-0.5 sm:flex-row sm:items-start sm:gap-3">
              <div className="min-w-0 flex-1">
                <p className="truncate text-sm font-medium text-foreground-emphasis">
                  {workflow.name ?? "Unnamed workflow"}
                </p>
                {workflow.description ? (
                  <p className="truncate text-body-sm leading-inherit text-foreground-tertiary">
                    {workflow.description}
                  </p>
                ) : null}
              </div>
              {workflow.schedule_cron ? (
                <span className="flex shrink-0 items-center gap-1 text-xs text-foreground-tertiary sm:mt-0.5">
                  <Icon icon={Calendar03Icon} size={13} aria-hidden="true" />
                  {getCadenceLabel(workflow.schedule_cron)}
                </span>
              ) : null}
            </div>
          </li>
        ))}
      </ul>
    </ExpertSection>
  );
}
