"use client";

import { Text } from "@/components/atoms/Text/Text";
import { AutopilotAvatar } from "@/components/molecules/AutopilotAvatar/AutopilotAvatar";
import { TypingText } from "@/components/molecules/TypingText/TypingText";
import {
  ChecklistMark,
  type ChecklistState,
} from "../components/ChecklistMark";
import { usePreparingStep } from "./usePreparingStep";

interface Props {
  onComplete: () => void;
}

export function PreparingStep({ onComplete }: Props) {
  const { started, progress, completedItems, checklist } = usePreparingStep({
    onComplete,
  });

  return (
    <div className="flex w-full max-w-md flex-col items-center gap-8 px-4">
      <div className="flex flex-col items-center gap-4">
        <AutopilotAvatar size={120} transparent />
        <Text variant="h4" className="text-center">
          <TypingText
            text="Preparing your workspace..."
            active={started}
            delay={400}
            speed={60}
          />
        </Text>
      </div>

      <div className="h-0.5 w-full overflow-hidden bg-surface-hover">
        <div
          className="h-full bg-action-hover transition-all duration-100 ease-linear"
          style={{ width: `${progress}%` }}
        />
      </div>

      <ul className="flex flex-col gap-3">
        {checklist.map((item, i) => {
          const state = getChecklistState(i, completedItems);
          return (
            <li key={item} className="flex items-center gap-3">
              <ChecklistMark key={state} state={state} />
              <Text
                variant={state === "active" ? "large-medium" : "large"}
                as="span"
                tone={state === "pending" ? "muted" : "primary"}
                className="transition-colors"
              >
                {item}
              </Text>
            </li>
          );
        })}
      </ul>
    </div>
  );
}

function getChecklistState(index: number, completedItems: number) {
  if (index < completedItems) return "done" satisfies ChecklistState;
  if (index === completedItems) return "active" satisfies ChecklistState;
  return "pending" satisfies ChecklistState;
}
