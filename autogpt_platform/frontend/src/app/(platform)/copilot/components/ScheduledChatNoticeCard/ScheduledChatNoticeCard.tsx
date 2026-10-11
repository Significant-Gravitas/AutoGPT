"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import {
  Clock01Icon,
  CreditCardIcon,
  PlugSocketIcon,
} from "@hugeicons/core-free-icons";
import type { IconSvgElement } from "@hugeicons/react";
import type { ScheduledChatNotice } from "./helpers";

interface Props {
  notice: ScheduledChatNotice;
}

const REASON_ICONS: Record<string, IconSvgElement> = {
  usage: Clock01Icon,
  no_plan: CreditCardIcon,
  codex_access: PlugSocketIcon,
};

export function ScheduledChatNoticeCard({ notice }: Props) {
  const icon = REASON_ICONS[notice.reason] ?? Clock01Icon;
  const showActions = notice.actions.length > 0;

  return (
    <div
      className="max-w-md rounded-2xl bg-white p-4 ring-1 ring-inset ring-zinc-200/60"
      data-testid="scheduled-chat-notice"
    >
      <div className="flex items-start gap-3">
        <Icon icon={icon} size={18} className="mt-0.5 shrink-0 text-zinc-400" />
        <div className="min-w-0">
          <p className="text-sm font-medium text-zinc-900">{notice.headline}</p>
          <p className="mt-1 text-sm text-zinc-500">{notice.body}</p>
          {showActions ? (
            <div className="mt-3 flex flex-wrap gap-2">
              {notice.actions.map((action) => (
                <Button
                  key={action.href}
                  as="NextLink"
                  href={action.href}
                  variant="secondary"
                  size="small"
                >
                  {action.label}
                </Button>
              ))}
            </div>
          ) : null}
        </div>
      </div>
    </div>
  );
}
