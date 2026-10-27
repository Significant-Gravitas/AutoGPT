"use client";

import { PauseIcon } from "@hugeicons/core-free-icons";

import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";

import type { TurnPause } from "../../../../turnPause";
import { PausedCardActions } from "./components/PausedCardActions";
import { useLivePausedCard } from "./useLivePausedCard";

interface Props {
  pause: TurnPause;
  sessionId: string;
  assistantName: string;
  onFollow: () => void;
  onStop: () => void;
}

export function LivePausedCard({ onStop, ...args }: Props) {
  const card = useLivePausedCard(args);
  const hasActions =
    card.resume.resumesAt !== null ||
    card.primary !== null ||
    card.chatGptName !== null ||
    card.canStop ||
    card.canUpgrade;

  return (
    <section
      aria-label={card.title}
      className="my-2 overflow-hidden rounded-2xl border border-border-base bg-surface shadow-sm duration-500 animate-in fade-in slide-in-from-bottom-2 fill-mode-both"
    >
      <div className="flex items-start gap-3 px-5 py-4">
        <span
          // TODO(tokens): no exact role for border-purple-200, bg-purple-50; no fitting role for text-purple-600 #6C2EDF; #6C2EDF is brand-hover (a fill)
          className="flex size-9 shrink-0 items-center justify-center rounded-xl border border-purple-200 bg-purple-50 text-purple-600"
        >
          <Icon icon={PauseIcon} size={18} />
        </span>
        <div className="min-w-0 flex-1">
          <Text variant="body-medium" className="!text-foreground-strong">
            {card.title}
          </Text>
          <Text
            variant="small"
            className="mt-1 !leading-5 !text-foreground-tertiary"
          >
            {card.body}
          </Text>
        </div>
      </div>
      {hasActions && <PausedCardActions card={card} onStop={onStop} />}
    </section>
  );
}
