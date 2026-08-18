import { Icon } from "@/components/atoms/Icon/Icon";
import { SparklesIcon } from "@hugeicons/core-free-icons";
import { ExpertAccent } from "../../../../helpers";

interface Props {
  expertName: string;
  accent: ExpertAccent;
}

export function IncludedPlanSection({ expertName, accent }: Props) {
  return (
    <div className="relative mt-8 flex flex-col gap-2 rounded-xl bg-zinc-50/80 px-4 py-3.5">
      <div className="flex items-center gap-2 text-sm font-medium text-zinc-700">
        <Icon icon={SparklesIcon} size={16} className={accent.icon} />
        Included with your plan
      </div>
      <p className="text-[13px] leading-relaxed text-zinc-500">
        {`${expertName} is an AI teammate. They'll always tell you before acting outside the platform.`}
      </p>
    </div>
  );
}
