"use client";

import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { BotAvatar } from "@/components/molecules/BotAvatar/BotAvatar";
import { AUTOPILOT_AVATAR } from "@/components/molecules/BotAvatar/helpers";
import { TypingText } from "@/components/molecules/TypingText/TypingText";
import { cn } from "@/lib/utils";
import { Tick02Icon } from "@hugeicons/core-free-icons";
import { usePreparingStep } from "./usePreparingStep";

interface Props {
  onComplete: () => void;
  isBrainDumpEnabled?: boolean;
}

export function PreparingStep({
  onComplete,
  isBrainDumpEnabled = false,
}: Props) {
  const { started, progress, completedItems, checklist } = usePreparingStep({
    onComplete,
    isBrainDumpEnabled,
  });

  return (
    <div className="flex w-full max-w-md flex-col items-center gap-8 px-4">
      <div className="flex flex-col items-center gap-4">
        <BotAvatar
          config={AUTOPILOT_AVATAR}
          status="working"
          size={120}
          trackPointer
          showBadge={false}
        />
        <Text variant="h4" className="text-center">
          <TypingText
            text="Preparing your workspace..."
            active={started}
            delay={400}
            speed={60}
          />
        </Text>
      </div>

      <div className="h-0.5 w-full overflow-hidden bg-zinc-100">
        <div
          className="h-full bg-zinc-900 transition-all duration-100 ease-linear"
          style={{ width: `${progress}%` }}
        />
      </div>

      <ul className="flex flex-col gap-2">
        {checklist.map((item, i) => (
          <li key={item} className="flex items-center gap-2">
            <Icon
              icon={Tick02Icon}
              size={16}
              className={cn(
                "shrink-0 transition-colors",
                i < completedItems ? "text-zinc-900" : "text-zinc-300",
              )}
            />
            <Text
              variant="body"
              as="span"
              tone={i < completedItems ? "primary" : "muted"}
              className="transition-colors"
            >
              {item}
            </Text>
          </li>
        ))}
      </ul>
    </div>
  );
}
