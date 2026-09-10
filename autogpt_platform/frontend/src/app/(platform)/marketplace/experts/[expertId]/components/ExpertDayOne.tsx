import { ExpertWorkflowRef } from "@/app/api/__generated__/models/expertWorkflowRef";
import { Icon } from "@/components/atoms/Icon/Icon";
import { cn } from "@/lib/utils";
import { FlashIcon } from "@hugeicons/core-free-icons";
import type { ExpertAccent } from "../../../components/ExpertsSection/helpers";
import { ExpertSection } from "./ExpertSection";

interface Props {
  name: string;
  workflow: ExpertWorkflowRef | null;
  accent: ExpertAccent;
}

export function ExpertDayOne({ name, workflow, accent }: Props) {
  if (!workflow) return null;

  return (
    <ExpertSection
      title={`What ${name} sets up on day one`}
      description="Running for you from the first conversation, with nothing to build."
    >
      <div
        className={cn(
          "flex items-start gap-3 rounded-xl border border-zinc-200/80 px-4 py-3.5",
          accent.wash,
        )}
      >
        <span className="mt-0.5 flex size-8 shrink-0 items-center justify-center rounded-lg bg-white ring-1 ring-inset ring-zinc-200/70">
          <Icon icon={FlashIcon} size={16} className={accent.icon} />
        </span>
        <div className="min-w-0">
          <div className="text-[15px] font-medium text-zinc-900">
            {workflow.name}
          </div>
          {workflow.description ? (
            <p className="mt-0.5 text-[13px] leading-5 text-zinc-600">
              {workflow.description}
            </p>
          ) : null}
        </div>
      </div>
    </ExpertSection>
  );
}
