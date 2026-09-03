import {
  AlertCircleIcon,
  CoinsDollarIcon,
  MessageQuestionIcon,
  PauseIcon,
  Settings02Icon,
} from "@hugeicons/core-free-icons";
import type { IconSvgElement } from "@hugeicons/react";
import { useState } from "react";
import type { HomeAttentionItem } from "@/app/api/__generated__/models/homeAttentionItem";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { ExpertAvatar } from "@/components/molecules/ExpertAvatar/ExpertAvatar";
import { AttentionRowActions } from "./AttentionRowActions";

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

const KIND_LABELS: Record<HomeAttentionItem["kind"], string> = {
  approval: "Approval",
  setup: "Setup",
  paused: "Paused",
  credits: "Credits",
  question: "Question",
};

export function AttentionRow({ item, isProcessing, onDecision }: Props) {
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
    <article className="flex flex-col gap-3 px-4 py-3 sm:flex-row sm:items-center">
      <div className="flex min-w-0 flex-1 items-start gap-3">
        {item.expert ? (
          <ExpertAvatar
            name={item.expert.name}
            avatarUrl={item.expert.avatar_url}
            size={24}
            className="mt-0.5"
          />
        ) : (
          <span className="mt-0.5 flex size-6 shrink-0 items-center justify-center rounded-md bg-zinc-100 text-zinc-500">
            <Icon icon={ICONS[item.kind]} size={13} aria-hidden="true" />
          </span>
        )}
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-2">
            <Text
              variant="body-medium"
              className="text-pretty text-[13px] leading-5 text-zinc-900"
            >
              {item.title}
            </Text>
            {item.priority === "high" ? (
              <span className="rounded bg-amber-50 px-1.5 py-px text-[10px] font-medium text-amber-700 ring-1 ring-inset ring-amber-600/10">
                Waiting
              </span>
            ) : null}
          </div>
          <Text
            variant="small"
            className="mt-0.5 line-clamp-2 text-pretty text-[13px] leading-5 text-zinc-500"
          >
            {item.description}
          </Text>
          <div className="mt-1 flex min-w-0 flex-wrap items-center gap-x-1.5 text-[11px] text-zinc-400">
            <span className="font-medium text-zinc-500">
              {item.expert?.name ?? KIND_LABELS[item.kind]}
            </span>
            <span aria-hidden="true">·</span>
            <span className="line-clamp-1 min-w-0">{item.why_it_matters}</span>
          </div>
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
