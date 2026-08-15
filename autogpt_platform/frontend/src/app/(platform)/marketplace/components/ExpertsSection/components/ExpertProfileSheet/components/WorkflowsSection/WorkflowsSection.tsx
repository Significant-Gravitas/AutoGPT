import { ExpertWorkflowRef } from "@/app/api/__generated__/models/expertWorkflowRef";
import { Icon } from "@/components/atoms/Icon/Icon";
import { cn } from "@/lib/utils";
import { FlashIcon } from "@hugeicons/core-free-icons";
import { ExpertAccent } from "../../../../helpers";
import { ProfileSection } from "../ProfileSection/ProfileSection";

interface Props {
  firstName: string;
  workflows: ExpertWorkflowRef[];
  accent: ExpertAccent;
}

export function WorkflowsSection({ firstName, workflows, accent }: Props) {
  if (!workflows.length) return null;

  return (
    <ProfileSection title={`Workflows ${firstName} brings`}>
      <div className="divide-y divide-zinc-100 rounded-xl border border-zinc-200/80 bg-white">
        {workflows.map((workflow) => (
          <div key={workflow.id} className="flex items-center gap-3 px-4 py-3">
            <Icon
              icon={FlashIcon}
              size={18}
              className={cn("shrink-0", accent.icon)}
            />
            <div className="min-w-0">
              <div className="text-[15px] font-medium text-zinc-800">
                {workflow.name ?? "Unnamed workflow"}
              </div>
              {workflow.description ? (
                <div className="line-clamp-1 text-[13px] text-zinc-500">
                  {workflow.description}
                </div>
              ) : null}
            </div>
          </div>
        ))}
      </div>
    </ProfileSection>
  );
}
