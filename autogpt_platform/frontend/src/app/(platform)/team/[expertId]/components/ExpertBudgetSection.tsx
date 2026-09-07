import { Expert } from "@/app/api/__generated__/models/expert";
import { Text } from "@/components/atoms/Text/Text";
import { creditsToUsdLabel } from "@/lib/credits";
import { SpendMeter } from "../../components/ExpertTeamCard/components/SpendMeter";
import { getWeeklySpend } from "../../helpers";

interface Props {
  expert: Expert;
}

export function ExpertBudgetSection({ expert }: Props) {
  const weeklySpend = getWeeklySpend(expert);

  return (
    <section
      aria-label={`${expert.name} budget`}
      className="flex w-full flex-col gap-1.5 lg:w-1/2"
    >
      <div className="flex items-baseline justify-between gap-2">
        <Text variant="body-medium" className="text-sm text-zinc-700">
          Budget
        </Text>
        <Text
          variant="body-medium"
          unmask={false}
          className="text-sm tabular-nums text-zinc-700"
        >
          {weeklySpend
            ? `${creditsToUsdLabel(weeklySpend.spent)} / ${creditsToUsdLabel(weeklySpend.budget)}`
            : "No budget"}
        </Text>
      </div>
      <SpendMeter
        spent={weeklySpend?.spent ?? 0}
        budget={weeklySpend?.budget ?? 1}
        color={expert.color}
        muted={!weeklySpend}
      />
    </section>
  );
}
