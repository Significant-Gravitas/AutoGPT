import { Expert } from "@/app/api/__generated__/models/expert";
import { Icon } from "@/components/atoms/Icon/Icon";
import { cn } from "@/lib/utils";
import { FlashIcon } from "@hugeicons/core-free-icons";
import type { ExpertAccent } from "../../../components/ExpertsSection/helpers";
import { ExpertSectionLabel } from "./ExpertSectionLabel";

interface Props {
  workflows: Expert["workflows"];
  accent: ExpertAccent;
}

export function ExpertWorkflowList({ workflows, accent }: Props) {
  if (workflows.length === 0) return null;

  return (
    <section>
      <ExpertSectionLabel>Preloaded workflows</ExpertSectionLabel>
      <div className="divide-y divide-zinc-100 rounded-2xl border border-zinc-200/80 bg-white">
        {workflows.map((workflow) => (
          <div key={workflow.id} className="flex items-start gap-3 px-4 py-3.5">
            <Icon
              icon={FlashIcon}
              size={18}
              className={cn("mt-0.5 shrink-0", accent.icon)}
            />
            <div className="min-w-0">
              <div className="text-[15px] font-medium text-zinc-800">
                {workflow.name ?? "Unnamed workflow"}
              </div>
              {workflow.description ? (
                <div className="line-clamp-2 text-sm text-zinc-500">
                  {workflow.description}
                </div>
              ) : null}
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}
