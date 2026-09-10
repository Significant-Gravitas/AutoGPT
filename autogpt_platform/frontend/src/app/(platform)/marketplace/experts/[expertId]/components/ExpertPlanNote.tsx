import { Icon } from "@/components/atoms/Icon/Icon";
import { creditsToUsdLabel } from "@/lib/credits";
import { cn } from "@/lib/utils";
import { SparklesIcon } from "@hugeicons/core-free-icons";
import type { ExpertAccent } from "../../../components/ExpertsSection/helpers";

interface Props {
  name: string;
  weeklyBudget: number | null;
  accent: ExpertAccent;
}

export function ExpertPlanNote({ name, weeklyBudget, accent }: Props) {
  return (
    <section className="flex items-start gap-3 rounded-xl bg-zinc-50 px-4 py-3.5">
      <Icon
        icon={SparklesIcon}
        size={16}
        className={cn("mt-0.5 shrink-0", accent.icon)}
      />
      <div>
        <h2 className="text-sm font-medium text-zinc-900">
          Included with your plan
        </h2>
        <p className="mt-0.5 text-[13px] leading-5 text-zinc-600">
          {`Hiring ${name} costs nothing extra. Their work draws on your existing balance`}
          {weeklyBudget
            ? `, capped at ${creditsToUsdLabel(weeklyBudget)} a week until you change it.`
            : "."}
        </p>
      </div>
    </section>
  );
}
