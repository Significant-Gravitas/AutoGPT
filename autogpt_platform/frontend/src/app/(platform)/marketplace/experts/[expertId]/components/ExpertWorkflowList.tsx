import { Expert } from "@/app/api/__generated__/models/expert";
import { Icon } from "@/components/atoms/Icon/Icon";
import { FlashIcon } from "@hugeicons/core-free-icons";
import type { ExpertAccent } from "../../../components/ExpertsSection/helpers";
import { ExpertSection } from "./ExpertSection";

interface Props {
  name: string;
  workflows: Expert["workflows"];
  accent: ExpertAccent;
}

export function ExpertWorkflowList({ name, workflows, accent }: Props) {
  if (workflows.length === 0) return null;

  return (
    <ExpertSection
      title="Workflows"
      count={workflows.length}
      description={`Installed and ready to run the moment ${name} joins your team.`}
    >
      <ul className="divide-y divide-zinc-100 overflow-hidden rounded-xl border border-zinc-200 bg-white">
        {workflows.map((workflow) => (
          <li key={workflow.id} className="flex items-start gap-3 px-4 py-3">
            <span className="mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-md bg-zinc-50 ring-1 ring-inset ring-zinc-200/70">
              <Icon icon={FlashIcon} size={14} className={accent.icon} />
            </span>
            <div className="min-w-0">
              <div className="text-sm font-medium text-zinc-900">
                {workflow.name ?? "Unnamed workflow"}
              </div>
              {workflow.description ? (
                <p className="mt-0.5 line-clamp-2 text-[13px] leading-5 text-zinc-500">
                  {workflow.description}
                </p>
              ) : null}
            </div>
          </li>
        ))}
      </ul>
    </ExpertSection>
  );
}
