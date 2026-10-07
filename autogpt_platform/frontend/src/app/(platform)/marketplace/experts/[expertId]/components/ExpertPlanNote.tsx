import { creditsToUsdLabel } from "@/lib/credits";
import { Coins01Icon } from "@hugeicons/core-free-icons";
import { ExpertNoteCard } from "./ExpertNoteCard";
import { ExpertSection } from "./ExpertSection";

interface Props {
  name: string;
  weeklyBudget: number | null;
}

export function ExpertPlanNote({ name, weeklyBudget }: Props) {
  return (
    <ExpertSection title="Included with your plan">
      <ExpertNoteCard icon={Coins01Icon}>
        <p className="text-base leading-7 text-zinc-600">
          {`Hiring ${name} costs nothing extra. Their work draws on your existing balance`}
          {weeklyBudget
            ? `, capped at ${creditsToUsdLabel(weeklyBudget)} a week — a limit you can change once they are on your team.`
            : "."}
        </p>
      </ExpertNoteCard>
    </ExpertSection>
  );
}
