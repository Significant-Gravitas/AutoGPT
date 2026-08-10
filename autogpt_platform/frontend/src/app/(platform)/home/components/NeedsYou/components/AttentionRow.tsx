import {
  AlertCircleIcon,
  Cancel01Icon,
  CoinsDollarIcon,
  PauseIcon,
  Settings02Icon,
  Tick02Icon,
} from "@hugeicons/core-free-icons";
import type { IconSvgElement } from "@hugeicons/react";
import { useState } from "react";
import type { HomeAttentionItem } from "@/app/api/__generated__/models/homeAttentionItem";
import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { ExpertAvatar } from "@/components/molecules/ExpertAvatar/ExpertAvatar";
import { cn } from "@/lib/utils";

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
};

const KIND_LABELS: Record<HomeAttentionItem["kind"], string> = {
  approval: "Approval",
  setup: "Setup",
  paused: "Paused",
  credits: "Credits",
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
    <article className="px-4 py-4 sm:px-5">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-start">
        <div className="flex min-w-0 flex-1 items-start gap-3">
          {item.expert ? (
            <ExpertAvatar
              name={item.expert.name}
              avatarUrl={item.expert.avatar_url}
            />
          ) : (
            <span className="flex size-9 shrink-0 items-center justify-center rounded-lg bg-zinc-50 text-zinc-500 ring-1 ring-inset ring-zinc-200">
              <Icon icon={ICONS[item.kind]} size={17} aria-hidden="true" />
            </span>
          )}
          <div className="min-w-0 flex-1">
            <div className="flex flex-wrap items-center gap-2">
              <Text variant="body-medium" className="text-pretty text-zinc-950">
                {item.title}
              </Text>
              {item.priority === "high" ? (
                <span className="rounded-md bg-amber-50 px-1.5 py-0.5 text-[10px] font-semibold text-amber-700">
                  Waiting
                </span>
              ) : null}
            </div>
            <Text
              variant="large"
              className="mt-1 line-clamp-2 text-pretty text-zinc-600"
            >
              {item.description}
            </Text>
            <div className="mt-1.5 flex min-w-0 flex-wrap items-center gap-x-1.5 text-xs text-zinc-400">
              <span className="font-medium text-zinc-500">
                {item.expert?.name ?? KIND_LABELS[item.kind]}
              </span>
              <span aria-hidden="true">·</span>
              <span className="line-clamp-1 min-w-0">
                {item.why_it_matters}
              </span>
            </div>
          </div>
        </div>

        <div className="flex shrink-0 items-center gap-2 self-end sm:self-center">
          {item.kind === "approval" && item.review ? (
            <>
              <Button
                as="NextLink"
                href={item.primary_action.href}
                variant="secondary"
                size="small"
              >
                {item.primary_action.label}
              </Button>
              <Button
                variant="icon"
                size="icon"
                className="size-10 border-zinc-800 bg-zinc-800 p-0 text-white hover:border-zinc-900 hover:bg-zinc-900"
                disabled={isProcessing}
                aria-label={`Approve: ${item.title}`}
                onClick={() => onDecision(item, true)}
              >
                <Icon icon={Tick02Icon} size={18} aria-hidden="true" />
              </Button>
              <Button
                variant="icon"
                size="icon"
                className={cn(
                  "size-10 p-0",
                  confirmDecline &&
                    "border-red-500 bg-red-500 text-white hover:border-red-600 hover:bg-red-600",
                )}
                disabled={isProcessing}
                aria-label={`${confirmDecline ? "Confirm decline" : "Decline"}: ${item.title}`}
                onClick={handleDecline}
                onBlur={() => setConfirmDecline(false)}
              >
                <Icon icon={Cancel01Icon} size={18} aria-hidden="true" />
              </Button>
            </>
          ) : (
            <Button
              as="NextLink"
              href={item.primary_action.href}
              variant="secondary"
              size="small"
            >
              {item.primary_action.label}
            </Button>
          )}
        </div>
      </div>
      <span className="sr-only" aria-live="polite">
        {confirmDecline ? `Press again to decline ${item.title}` : ""}
      </span>
    </article>
  );
}
