"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { AttentionCard } from "@/components/organisms/AttentionCard/AttentionCard";
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
    <AttentionCard.Root
      variant="outline"
      density="comfortable"
      className="max-w-md"
      data-testid="scheduled-chat-notice"
    >
      <AttentionCard.Row align="start">
        <AttentionCard.Icon nudge>
          <Icon icon={icon} size={18} />
        </AttentionCard.Icon>
        <AttentionCard.Content>
          <Text variant="body-medium" tone="primary" className="leading-5">
            {notice.headline}
          </Text>
          <Text variant="body" tone="muted" className="mt-1 leading-5">
            {notice.body}
          </Text>
          {showActions ? (
            <AttentionCard.Actions placement="below">
              {notice.actions.map(renderAction)}
            </AttentionCard.Actions>
          ) : null}
        </AttentionCard.Content>
      </AttentionCard.Row>
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
    </AttentionCard.Root>
  );
}
