import { Text } from "@/components/atoms/Text/Text";

interface Props {
  title: string;
  goal: string;
  previousGoal?: string | null;
}

/** The goal in one line, when it says more than the title, and a note when
 *  it changed on the way. */
export function GoalLine({ title, goal, previousGoal }: Props) {
  const showGoal = goal.trim() && goal.trim() !== title.trim();
  if (!showGoal && !previousGoal) return null;

  return (
    <div className="flex flex-col gap-1">
      {showGoal ? (
        <Text variant="inbox" tone="text" className="text-pretty">
          {goal}
        </Text>
      ) : null}
      {previousGoal ? (
        <Text variant="small" tone="faint">
          Goal updated · was “{previousGoal}”
        </Text>
      ) : null}
    </div>
  );
}
