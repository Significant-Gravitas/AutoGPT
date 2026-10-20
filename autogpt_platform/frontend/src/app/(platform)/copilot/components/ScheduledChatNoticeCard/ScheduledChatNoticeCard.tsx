"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { ConnectServiceDialog } from "@/components/contextual/IntegrationsPanel/components/ConnectServiceDialog/ConnectServiceDialog";
import {
  Alert01Icon,
  CheckmarkCircle02Icon,
  Clock01Icon,
  CreditCardIcon,
  PauseIcon,
  PlugSocketIcon,
} from "@hugeicons/core-free-icons";
import type { IconSvgElement } from "@hugeicons/react";
import { RunLessOftenDialog } from "./components/RunLessOftenDialog/RunLessOftenDialog";
import type { NoticeAction, ScheduledChatNotice } from "./helpers";
import { useScheduledChatNoticeCard } from "./useScheduledChatNoticeCard";

interface Props {
  notice: ScheduledChatNotice;
  sessionID?: string | null;
}

const REASON_ICONS: Record<string, IconSvgElement> = {
  usage: Clock01Icon,
  no_plan: CreditCardIcon,
  codex_access: PlugSocketIcon,
  connection: PlugSocketIcon,
  chat_link: PlugSocketIcon,
  budget: PauseIcon,
  error: Alert01Icon,
  resumed: CheckmarkCircle02Icon,
};

export function ScheduledChatNoticeCard({ notice, sessionID = null }: Props) {
  const card = useScheduledChatNoticeCard({
    scheduleKey: notice.scheduleKey,
    sessionID,
  });
  const icon = REASON_ICONS[notice.reason] ?? Clock01Icon;
  const showActions = notice.actions.length > 0;

  function renderAction(action: NoticeAction) {
    if (action.kind === "link") {
      return (
        <Button
          key={`link-${action.href}`}
          as="NextLink"
          href={action.href}
          variant="secondary"
          size="small"
        >
          {action.label}
        </Button>
      );
    }
    if (action.kind === "run_less_often") {
      return (
        <Button
          key="run-less-often"
          variant="secondary"
          size="small"
          onClick={() => card.setLessOftenOpen(true)}
        >
          {action.label}
        </Button>
      );
    }
    if (action.kind === "reconnect") {
      return (
        <Button
          key={`reconnect-${action.provider}`}
          variant="secondary"
          size="small"
          onClick={() => card.openConnect(action.provider)}
        >
          {action.label}
        </Button>
      );
    }
    return (
      <Button
        key="resume"
        variant="secondary"
        size="small"
        loading={card.isResuming}
        disabled={card.resumed}
        onClick={card.resume}
      >
        {card.resumed ? "Resumed" : action.label}
      </Button>
    );
  }

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
              {notice.actions.map(renderAction)}
            </div>
          ) : null}
        </div>
      </div>
      {notice.scheduleKey && card.lessOftenOpen ? (
        <RunLessOftenDialog
          scheduleKey={notice.scheduleKey}
          isOpen
          setIsOpen={card.setLessOftenOpen}
        />
      ) : null}
      {card.connectProvider ? (
        <ConnectServiceDialog
          open
          onOpenChange={(open) => {
            if (!open) card.closeConnect();
          }}
          initialProviderId={card.connectProvider}
          onConnected={card.handleConnected}
        />
      ) : null}
    </div>
  );
}
