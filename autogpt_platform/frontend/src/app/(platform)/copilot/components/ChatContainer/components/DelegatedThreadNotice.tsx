"use client";

import { ArrowTurnBackwardIcon } from "@hugeicons/core-free-icons";
import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { AUTOPILOT_NAME } from "@/components/molecules/AutopilotAvatar/helpers";
import { getSentFromDisplayName, type SentFrom } from "../../../sentFrom";
import { useExpertMap } from "../../../useExpertMap";

interface Props {
  sentFrom: SentFrom;
  expertName: string | null;
}

/** An expert thread that Otto (or a teammate) opened by delegation says so
 *  at the top, with the way back to the thread that owns the outcome. */
export function DelegatedThreadNotice({ sentFrom, expertName }: Props) {
  const { expertsById } = useExpertMap();
  const resolved = sentFrom.expertId
    ? expertsById.get(sentFrom.expertId)?.name
    : null;
  const from = getSentFromDisplayName(sentFrom, resolved);
  const isOtto = from === AUTOPILOT_NAME;

  return (
    <div
      data-testid="delegated-thread-notice"
      className="mx-auto flex w-full max-w-3xl items-center justify-between gap-3 px-3 pt-3"
    >
      <p className="min-w-0 truncate text-xs text-zinc-500">
        Delegated by {from}
        {expertName ? ` · ${expertName} is working for ${from}` : ""}
        {isOtto ? " · you own the outcome" : ""}
      </p>
      <Button
        as="NextLink"
        href={`/copilot?sessionId=${sentFrom.sessionId}`}
        variant="secondary"
        size="xs"
        leftIcon={<Icon icon={ArrowTurnBackwardIcon} size={12} />}
        className="shrink-0"
      >
        Back to {from}&apos;s thread
      </Button>
    </div>
  );
}
