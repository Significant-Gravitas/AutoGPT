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
        <span className="flex size-9 shrink-0 items-center justify-center rounded-xl border border-brand/50 bg-accent-surface text-brand-hover">
          <Icon icon={PauseIcon} size={18} />
        </span>
        <div className="min-w-0 flex-1">
          <Text tone="primary" variant="body-medium">
            {card.title}
          </Text>
          <Text tone="muted" variant="small" className="mt-1 !leading-5">
            {card.body}
          </Text>
        </div>
      </div>
      {hasActions && <PausedCardActions card={card} onStop={onStop} />}
    </section>
  );
}
