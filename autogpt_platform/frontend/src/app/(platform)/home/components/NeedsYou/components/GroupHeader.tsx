import { useState } from "react";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { AUTOPILOT_NAME } from "@/components/molecules/AutopilotAvatar/helpers";
import { approveAllLabel } from "@/app/(platform)/copilot/components/ApprovalQueue/helpers";
import type { AttentionGroup } from "../helpers";
import { HeldAvatar } from "./HeldAvatar";

interface Props {
  group: AttentionGroup;
  pending: number;
  rejectable: number;
  approvable: number;
  busy: boolean;
  onRejectAll: () => void;
  onApproveAll: () => void;
}

export function GroupHeader({
  group,
  pending,
  rejectable,
  approvable,
  busy,
  onRejectAll,
  onApproveAll,
}: Props) {
  const [confirming, setConfirming] = useState(false);
  const name = group.expert?.name ?? AUTOPILOT_NAME;

  return (
    <div className="flex flex-wrap items-center gap-2 bg-zinc-50/60 px-4 py-2">
      <HeldAvatar item={group.rows[0].item} size={24} />
      <Text variant="body-medium" as="h3" tone="primary">
        {name}
      </Text>
      {pending > 0 && (
        <Text
          variant="body"
          as="span"
          tone="secondary"
          className="tabular-nums"
          aria-label={`${pending} waiting`}
        >
          {pending}
        </Text>
      )}
      <div className="ml-auto flex flex-wrap items-center gap-2 text-sm">
        {confirming ? (
          <>
            <span role="alert" className="text-zinc-700">
              Reject all {rejectable}? {name} will be told none of them ran.
            </span>
            <Button
              size="xs"
              variant="primary"
              className="rounded-full"
              disabled={busy}
              onClick={() => {
                setConfirming(false);
                onRejectAll();
              }}
            >
              Reject all
            </Button>
            <Button
              size="xs"
              variant="ghost"
              className="rounded-full"
              onClick={() => setConfirming(false)}
            >
              Cancel
            </Button>
          </>
        ) : (
          <>
            {approvable >= 2 && (
              <Button
                size="xs"
                variant="secondary"
                className="rounded-full"
                disabled={busy}
                onClick={onApproveAll}
              >
                {approveAllLabel(approvable)}
              </Button>
            )}
            {rejectable >= 2 && (
              <Button
                variant="link"
                size="small"
                className="h-auto min-w-0 px-0 py-0 text-sm"
                disabled={busy}
                onClick={() => setConfirming(true)}
              >
                Reject all {rejectable}…
              </Button>
            )}
          </>
        )}
      </div>
    </div>
  );
}
