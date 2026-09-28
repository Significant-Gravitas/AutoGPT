"use client";

import {
  ArrowLeft01Icon,
  ArrowReloadHorizontalIcon,
  ArrowRight01Icon,
  File02Icon,
} from "@hugeicons/core-free-icons";
import { useContext } from "react";
import { Badge } from "@/components/atoms/Badge/Badge";
import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { AutopilotAvatar } from "@/components/molecules/AutopilotAvatar/AutopilotAvatar";
import { ExpertAvatar } from "@/components/molecules/ExpertAvatar/ExpertAvatar";
import {
  type ChatDelegation,
  delegationName,
  getDelegationStatusView,
} from "../../../../../delegations";
import { useDelegationLive } from "../../../../../useDelegationLive";
import { CopilotChatActionsContext } from "../../../../CopilotChatActionsProvider/useCopilotChatActions";
import { retryMessage } from "../../../../DelegationStatusLine/helpers";
import {
  BADGE_VARIANT,
  delegationSubtitle,
  delegationTitle,
  threadHref,
} from "../helpers";
import { DelegationTimeline } from "./DelegationTimeline";

interface Props {
  delegation: ChatDelegation;
  onBack: () => void;
}

function Section({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
  return (
    <div className="flex flex-col gap-1.5">
      <Text variant="eyebrow">{title}</Text>
      {children}
    </div>
  );
}

export function DelegationDetail({ delegation, onBack }: Props) {
  const live = useDelegationLive(delegation);
  const actions = useContext(CopilotChatActionsContext);
  const view = getDelegationStatusView(live.status);
  const name = delegationName(delegation);
  const href = threadHref(delegation);
  const subtitle = delegationSubtitle(live.status, live.elapsedSeconds);

  return (
    <div className="flex flex-col gap-4 p-4" data-testid="delegation-detail">
      <button
        type="button"
        onClick={onBack}
        className="flex w-fit items-center gap-1 text-xs text-zinc-600 hover:text-zinc-900"
      >
        <Icon icon={ArrowLeft01Icon} size={14} />
        All work
      </button>
      <div className="flex items-start justify-between gap-2">
        <div className="flex min-w-0 items-center gap-2.5">
          <span className="flex shrink-0 items-center gap-1">
            <AutopilotAvatar size={24} />
            <Icon icon={ArrowRight01Icon} size={12} className="text-zinc-400" />
            <ExpertAvatar
              name={name}
              avatarUrl={delegation.expert?.avatarUrl ?? null}
              size={24}
            />
          </span>
          <span className="flex min-w-0 flex-col">
            <Text variant="h5" className="truncate">
              {delegationTitle(delegation)}
            </Text>
            {subtitle && (
              <span className="text-xs text-zinc-500">{subtitle}</span>
            )}
          </span>
        </div>
        <Badge variant={BADGE_VARIANT[view.tone]} className="shrink-0">
          {view.label}
        </Badge>
      </div>

      {live.status === "needs-input" && live.question && (
        <Section title={`${name} asks`}>
          <div className="flex flex-col gap-2 rounded-xl border border-amber-200 bg-amber-50 p-3">
            <p className="text-sm text-zinc-900">{live.question}</p>
            {href && (
              <Button
                as="NextLink"
                href={href}
                variant="primary"
                size="xs"
                className="w-fit"
              >
                Answer in {name}&apos;s thread
              </Button>
            )}
          </div>
        </Section>
      )}

      {live.status === "failed" && (
        <Section title="What to do">
          {delegation.error && (
            <p className="text-sm text-red-700">{delegation.error}</p>
          )}
          <div className="flex flex-wrap gap-2">
            {actions && (
              <Button
                variant="primary"
                size="xs"
                leadingIcon={ArrowReloadHorizontalIcon}
                onClick={() => void actions.onSend(retryMessage(delegation))}
              >
                Retry
              </Button>
            )}
            {href && (
              <Button as="NextLink" href={href} variant="secondary" size="xs">
                Open thread
              </Button>
            )}
          </div>
        </Section>
      )}

      {live.status !== "failed" && href && (
        <div className="flex flex-wrap gap-2">
          <Button
            as="NextLink"
            href={href}
            variant="secondary"
            size="xs"
            rightIcon={<Icon icon={ArrowRight01Icon} size={12} />}
          >
            Open thread
          </Button>
        </div>
      )}

      {(delegation.response || delegation.files.length > 0) && (
        <Section title="What came back">
          {delegation.response && (
            <p className="line-clamp-6 whitespace-pre-wrap text-sm text-zinc-700">
              {delegation.response}
            </p>
          )}
          {delegation.files.map((file) => (
            <span
              key={file.path}
              className="flex items-center gap-1.5 text-sm text-zinc-700"
            >
              <Icon icon={File02Icon} size={14} className="text-zinc-500" />
              <span className="truncate">{file.name}</span>
            </span>
          ))}
        </Section>
      )}

      {delegation.prompt && (
        <Section title="Brief">
          <p className="line-clamp-8 whitespace-pre-wrap text-sm text-zinc-700">
            {delegation.prompt}
          </p>
        </Section>
      )}

      <Section title="Timeline">
        <DelegationTimeline steps={live.steps} latestText={live.latestText} />
      </Section>
    </div>
  );
}
