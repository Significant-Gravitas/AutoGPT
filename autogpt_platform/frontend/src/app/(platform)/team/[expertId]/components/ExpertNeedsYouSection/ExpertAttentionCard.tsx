"use client";

import { AttentionRowActions } from "@/app/(platform)/home/components/NeedsYou/components/AttentionRowActions";
import type { HomeAttentionItem } from "@/app/api/__generated__/models/homeAttentionItem";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { ExpertAvatar } from "@/components/molecules/ExpertAvatar/ExpertAvatar";
import {
  AlertCircleIcon,
  CoinsDollarIcon,
  MessageQuestionIcon,
  PauseIcon,
  Settings02Icon,
} from "@hugeicons/core-free-icons";
import type { IconSvgElement } from "@hugeicons/react";
import { useState } from "react";

interface Props {
  item: HomeAttentionItem;
  isProcessing: boolean;
  onDecision: (item: HomeAttentionItem, approved: boolean) => void;
}

const ICONS: Record<HomeAttentionItem["kind"], IconSvgElement> = {
  approval: AlertCircleIcon,
  setup: Settings02Icon,
  paused: PauseIcon,
  credits: CoinsDollarIcon,
  question: MessageQuestionIcon,
};

/** Compact card version of the home page's AttentionRow, styled like the
 *  chat sidebar's mini cards. */
export function ExpertAttentionCard({ item, isProcessing, onDecision }: Props) {
  const [confirmDecline, setConfirmDecline] = useState(false);

  function handleDecline() {
    if (!confirmDecline) {
      setConfirmDecline(true);
      return;
    }
    setConfirmDecline(false);
    onDecision(item, false);
  }

  return (
    <article className="flex flex-col gap-3 rounded-2xl bg-white px-3.5 py-2.5 smooth-shadow-ring-sm sm:flex-row sm:items-center">
      <div className="flex min-w-0 flex-1 items-center gap-2.5">
        {item.expert ? (
          <ExpertAvatar
            name={item.expert.name}
            avatarUrl={item.expert.avatar_url}
            size={32}
          />
        ) : (
          <Icon
            icon={ICONS[item.kind]}
            size={18}
            className="shrink-0 text-zinc-500"
            aria-hidden="true"
          />
        )}
        <div className="flex min-w-0 flex-1 flex-col">
          <div className="flex flex-wrap items-center gap-2">
            <Text
              variant="body-medium"
              as="span"
              className="truncate !text-zinc-800"
            >
              {item.title}
            </Text>
            {item.priority === "high" ? (
              <Text
                variant="small-medium"
                as="span"
                className="rounded bg-amber-50 px-1.5 py-px text-amber-700 ring-1 ring-inset ring-amber-600/10"
              >
                Waiting
              </Text>
            ) : null}
          </div>
          <Text variant="small" as="span" className="truncate !text-zinc-400">
            {item.description}
          </Text>
        </div>
      </div>

      <div className="flex shrink-0 items-center gap-1.5 self-end sm:self-center">
        <AttentionRowActions
          item={item}
          isProcessing={isProcessing}
          confirmDecline={confirmDecline}
          onApprove={() => onDecision(item, true)}
          onDecline={handleDecline}
          onDeclineBlur={() => setConfirmDecline(false)}
        />
      </div>
      <span className="sr-only" aria-live="polite">
        {confirmDecline ? `Press again to decline ${item.title}` : ""}
      </span>
    </article>
  );
}
