import { creditsToUsdLabel } from "@/lib/credits";
import { ExpertSection } from "./ExpertSection";

interface Props {
  name: string;
  weeklyBudget: number | null;
}

export function ExpertPlanNote({ name, weeklyBudget }: Props) {
  return (
    <ExpertSection title="Included with your plan">
      <p className="text-[15px] leading-6 text-zinc-600">
        {`Hiring ${name} costs nothing extra. Their work draws on your existing balance`}
        {weeklyBudget
          ? `, capped at ${creditsToUsdLabel(weeklyBudget)} a week — a limit you can change once they are on your team.`
          : "."}
      </p>
    </ExpertSection>
  );
}
