"use client";

import { AutoGPTLogo } from "@/components/atoms/AutoGPTLogo/AutoGPTLogo";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
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
        <AutoGPTLogo
          className="relative right-[3rem] h-24 w-[12rem]"
          hideText
        />
        <Text variant="h3" className="text-center">
          <TypingText
            text="Preparing your workspace..."
            active={started}
            delay={400}
            speed={60}
          />
        </Text>
      </div>

      <div className="h-2 w-full overflow-hidden rounded-full bg-gray-200">
        <div
          className="h-full rounded-full bg-purple-500 transition-all duration-100 ease-linear"
          style={{ width: `${progress}%` }}
        />
      </div>

      <ul className="flex flex-col gap-3">
        {checklist.map((item, i) => (
          <li key={item} className="flex items-center gap-3">
            <div
              className={cn(
                "flex h-6 w-6 items-center justify-center rounded-full transition-colors",
                i < completedItems
                  ? "bg-neutral-900 text-white"
                  : "bg-gray-200 text-gray-400",
              )}
            >
              <Icon icon={Tick02Icon} size={14} />
            </div>
            <Text
              variant="body"
              as="span"
              className={cn(
                "transition-colors",
                i < completedItems ? "!text-black" : "!text-zinc-500",
              )}
            >
              {item}
            </Text>
          </li>
        ))}
      </ul>
    </div>
  );
}
