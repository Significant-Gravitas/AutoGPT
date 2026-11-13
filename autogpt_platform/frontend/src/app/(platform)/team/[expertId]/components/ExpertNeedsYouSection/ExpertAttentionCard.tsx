"use client";

import { AttentionRowActions } from "@/app/(platform)/home/components/NeedsYou/components/AttentionRowActions";
import { HeldPassageQuote } from "@/app/(platform)/home/components/NeedsYou/components/HeldPassageQuote";
import { attentionReason } from "@/app/(platform)/copilot/components/ApprovalQueue/helpers";
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
  const reason = attentionReason(item.review);

  function handleDecline() {
    if (!confirmDecline) {
      setConfirmDecline(true);
      return;
    }
    setConfirmDecline(false);
    onDecision(item, false);
  }

  return (
    <article className="flex flex-col gap-3 rounded-2xl bg-surface px-3.5 py-2.5 smooth-shadow-ring-sm sm:flex-row sm:items-center">
      <div className="flex min-w-0 flex-1 items-center gap-2.5">
        {item.expert ? (
          <ExpertAvatar
            name={item.expert.name}
            avatarUrl={item.expert.avatar_url}
            size={32}
            // oxlint-disable-next-line design-system/no-restyle -- ExpertAvatar has no bordered variant (src/components is P9's)
            className="rounded-full border border-border-base"
          />
        ) : (
          <span className="inline-flex shrink-0 text-foreground-tertiary">
            <Icon icon={ICONS[item.kind]} size={18} aria-hidden="true" />
          </span>
        )}
        <div className="flex min-w-0 flex-1 flex-col">
          <div className="flex flex-wrap items-center gap-2">
            <Text
              variant="body-medium"
              as="span"
              tone="emphasis"
              className="break-words"
            >
              {item.title}
            </Text>
            {item.priority === "high" ? (
              <span className="rounded bg-warning-surface px-1.5 py-px font-sans text-caption font-medium text-warning-foreground ring-1 ring-warning-foreground/10 ring-inset">
                Waiting
              </span>
            ) : null}
          </div>
          <Text variant="small" as="span" tone="subtle" className="break-words">
            {reason?.line ?? item.description}
          </Text>
          {reason?.passage ? (
            <HeldPassageQuote passage={reason.passage} />
          ) : null}
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
