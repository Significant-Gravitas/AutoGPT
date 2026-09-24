"use client";

import { Button } from "@/components/atoms/Button/Button";
import { AUTOPILOT_NAME } from "@/components/molecules/AutopilotAvatar/helpers";
import { ApprovalFields } from "@/components/organisms/ApprovalFields/ApprovalFields";
import {
  type ApprovalItem,
  type ChatRule,
  isBare,
  isHeldRead,
  reasonLine,
} from "../../helpers";
import { ApprovalHeadline } from "../ApprovalHeadline";
import { ApproveSplitButton } from "./ApproveSplitButton";
import { HeldPassage } from "./HeldPassage";
import { MoneyBlock } from "./MoneyBlock";
import { useApprovalFields } from "./useApprovalFields";

export type CardStatus = "idle" | "approving" | "rejecting";

interface Props {
  item: ApprovalItem;
  status: CardStatus;
  failed: boolean;
  onApprove: (rule?: ChatRule) => void;
  onReject: () => void;
}

export function ApprovalCard({
  item,
  status,
  failed,
  onApprove,
  onReject,
}: Props) {
  const fields = useApprovalFields(item);
  const reason = reasonLine(item);
  const read = isHeldRead(item);
  const busy = status !== "idle";

  const actions = (
    <div className="grid grid-cols-2 gap-2 sm:flex sm:shrink-0 sm:flex-wrap">
      <ApproveSplitButton
        label={read ? `Release to ${AUTOPILOT_NAME}` : "Approve"}
        subjectName={item.subject.name}
        rules={read || item.spend ? [] : item.chatRulesAllowed}
        loading={status === "approving"}
        disabled={busy}
        onApprove={onApprove}
      />
      <Button
        size="small"
        variant="secondary"
        className="min-w-0"
        loading={status === "rejecting"}
        disabled={busy}
        onClick={onReject}
      >
        {status === "rejecting"
          ? "Rejecting…"
          : read
            ? "Keep it out"
            : "Reject"}
      </Button>
    </div>
  );

  // Nothing to read beyond the headline: the card is one row, as a list line is.
  if (isBare(item) && !failed) {
    return (
      <article
        aria-busy={busy}
        className="flex flex-col gap-2 px-4 py-3 sm:flex-row sm:items-center sm:gap-3"
      >
        <div className="min-w-0 flex-1">
          <ApprovalHeadline item={item} />
        </div>
        {actions}
      </article>
    );
  }

  return (
    <article aria-busy={busy} className="flex flex-col gap-2 px-4 py-3">
      <ApprovalHeadline item={item} />
      <div className="flex flex-col gap-3 sm:pl-[38px]">
        {failed && (
          <p role="alert" className="text-sm text-red-600">
            Couldn&apos;t send your answer. Nothing ran. Try again.
          </p>
        )}
        {reason && <p className="-mt-1 text-sm text-zinc-500">{reason}</p>}
        {item.spend && <MoneyBlock spend={item.spend} />}
        {read ? (
          item.passage && <HeldPassage passage={item.passage} />
        ) : (
          <ApprovalFields
            fields={fields.labels}
            values={fields.values}
            clipped={item.clipped}
            hiddenKeys={item.headlineKeys}
            idsWhenAlone={!item.headline.object}
          />
        )}
        {actions}
      </div>
    </article>
  );
}
