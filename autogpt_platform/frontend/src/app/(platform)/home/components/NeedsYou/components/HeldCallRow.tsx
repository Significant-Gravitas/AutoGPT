import { Cancel01Icon, Tick02Icon } from "@hugeicons/core-free-icons";
import { useState } from "react";
import type { HomeAttentionItem } from "@/app/api/__generated__/models/homeAttentionItem";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { AUTOPILOT_NAME } from "@/components/molecules/AutopilotAvatar/helpers";
import type { CardStatus } from "@/app/(platform)/copilot/components/ApprovalQueue/components/ApprovalCard/ApprovalCard";
import { HeadlineText } from "@/app/(platform)/copilot/components/ApprovalQueue/components/ApprovalHeadline";
import {
  type ApprovalItem,
  isHeldRead,
  modeLabel,
  reasonLine,
} from "@/app/(platform)/copilot/components/ApprovalQueue/helpers";
import { headlineButtonId, isInformed, shortAge } from "../helpers";
import { HeldAvatar } from "./HeldAvatar";
import { HeldPassageQuote } from "./HeldPassageQuote";

interface Props {
  item: HomeAttentionItem;
  approval: ApprovalItem;
  status: CardStatus;
  failed: boolean;
  avatarSize: number | null;
  onOpen: () => void;
  onDecide: (approved: boolean) => void;
}

export function HeldCallRow({
  item,
  approval,
  status,
  failed,
  avatarSize,
  onOpen,
  onDecide,
}: Props) {
  const [confirmReject, setConfirmReject] = useState(false);
  const read = isHeldRead(approval);
  const reason = reasonLine(approval);
  const mode = approval.reasonKind === "mode" ? modeLabel(approval.mode) : null;
  const busy = status !== "idle";
  const approveLabel = read ? "Release" : "Approve";
  const rejectLabel = read ? "Keep it out" : "Reject";

  function handleReject() {
    if (!confirmReject) {
      setConfirmReject(true);
      return;
    }
    setConfirmReject(false);
    onDecide(false);
  }

  return (
    <article
      aria-busy={busy}
      className="flex flex-col gap-3 px-4 py-3 sm:flex-row sm:items-center"
    >
      <div className="flex min-w-0 flex-1 items-start gap-3">
        {avatarSize ? <HeldAvatar item={item} size={avatarSize} /> : null}
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-x-2 gap-y-1">
            <button
              type="button"
              id={headlineButtonId(item.id)}
              aria-haspopup="dialog"
              onClick={onOpen}
              className="line-clamp-2 min-w-0 rounded-md text-left text-zinc-900 [overflow-wrap:anywhere] hover:opacity-80 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-zinc-300"
            >
              <Text variant="body-medium" as="span" className="text-pretty">
                <HeadlineText item={approval} />
              </Text>
            </button>
            {approval.subject.irreversible && (
              <Tag className="bg-red-50 text-red-700">Can&apos;t be undone</Tag>
            )}
            {mode && <Tag className="bg-zinc-100 text-zinc-600">{mode}</Tag>}
            {item.priority === "high" && (
              <Tag className="bg-amber-50 text-amber-700 ring-1 ring-inset ring-amber-600/10">
                Waiting
              </Tag>
            )}
          </div>
          {failed ? (
            <Text variant="body" role="alert" className="!text-red-600">
              Couldn&apos;t send your answer. Nothing ran. Try again.
            </Text>
          ) : reason ? (
            <Text
              variant="body"
              tone="secondary"
              className="text-pretty break-words"
            >
              {reason}
            </Text>
          ) : null}
          {read && approval.passage ? (
            <HeldPassageQuote
              passage={approval.passage}
              clamp={!isInformed(approval)}
            />
          ) : null}
        </div>
      </div>

      <div className="flex shrink-0 items-center gap-1.5 self-end sm:self-center">
        {item.created_at ? (
          <Text
            variant="small"
            as="span"
            tone="secondary"
            className="mr-1 hidden tabular-nums sm:inline"
          >
            <time dateTime={new Date(item.created_at).toISOString()}>
              {shortAge(item.created_at)}
            </time>
          </Text>
        ) : null}
        {isInformed(approval) ? (
          <Button
            variant="primary"
            size="icon-sm"
            leadingIcon={Tick02Icon}
            loading={status === "approving"}
            disabled={busy}
            aria-label={`${approveLabel}: ${item.title}`}
            onClick={() => onDecide(true)}
          />
        ) : (
          <Button
            variant="secondary"
            size="small"
            className="h-8 min-w-0 px-3"
            disabled={busy}
            aria-label={`Review: ${item.title}`}
            onClick={onOpen}
          >
            Review
          </Button>
        )}
        <Button
          variant={confirmReject ? "destructive" : "icon"}
          size="icon-sm"
          leadingIcon={Cancel01Icon}
          loading={status === "rejecting"}
          disabled={busy}
          aria-label={`${confirmReject ? `Confirm ${rejectLabel.toLowerCase()}` : rejectLabel}: ${item.title}`}
          onClick={handleReject}
          onBlur={() => setConfirmReject(false)}
        />
      </div>
      <span className="sr-only" aria-live="polite">
        {confirmReject
          ? `Press again to ${rejectLabel.toLowerCase()} ${item.title}. ${read ? approval.reader : AUTOPILOT_NAME} will be told it didn't run.`
          : ""}
      </span>
    </article>
  );
}

function Tag({
  className,
  children,
}: {
  className: string;
  children: React.ReactNode;
}) {
  return (
    <Text
      variant="small-medium"
      as="span"
      className={`shrink-0 rounded px-1.5 py-px ${className}`}
    >
      {children}
    </Text>
  );
}
