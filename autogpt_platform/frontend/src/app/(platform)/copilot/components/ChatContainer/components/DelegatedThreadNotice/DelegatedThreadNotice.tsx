"use client";

import { ArrowLeft01Icon } from "@hugeicons/core-free-icons";
import { Badge } from "@/components/atoms/Badge/Badge";
import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import type { SentFrom } from "../../../../sentFrom";
import { ExpectedBack } from "./components/ExpectedBack";
import { ThreadRoute } from "./components/ThreadRoute";
import { formatClock, getCapLine, getThreadTag } from "./helpers";
import { useDelegatedThreadNotice } from "./useDelegatedThreadNotice";

interface Props {
  sentFrom: SentFrom;
  sessionId: string;
  expert: {
    name: string;
    avatarUrl: string | null;
    color?: string | null;
  } | null;
  openingText: string | null;
}

const TAG_CLASS = {
  working: "bg-violet-50 text-violet-700 ring-violet-600/15",
  done: "",
  stopped: "",
} as const;

/** An expert thread opened by a delegation leads with the hand-off: who
 *  sent it, what it is for, what is expected back and the way home. */
export function DelegatedThreadNotice({
  sentFrom,
  sessionId,
  expert,
  openingText,
}: Props) {
  const { from, isOtto, delegator, delegation, capUsd } =
    useDelegatedThreadNotice({ sentFrom, sessionId });
  const tag = getThreadTag(delegation?.status ?? null, from);
  const clock = formatClock(delegation?.created_at);
  const brief = delegation?.brief || openingText;
  const subline = [
    delegation?.title ? `Part of “${delegation.title}”` : null,
    isOtto ? "you own the outcome" : null,
  ]
    .filter(Boolean)
    .join(" · ");

  return (
    <div className="mx-auto w-full max-w-3xl px-3 pt-3">
      <section
        data-testid="delegated-thread-notice"
        aria-label={`Delegated by ${from}`}
        className="flex flex-col gap-3 rounded-2xl border border-zinc-200 bg-white px-4 py-3.5"
      >
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div className="flex min-w-0 items-center gap-2.5">
            <ThreadRoute delegator={delegator} expert={expert} />
            <div className="flex min-w-0 flex-col">
              <Text variant="body-medium" tone="primary" className="truncate">
                Delegated by {from}
                {clock ? ` · ${clock}` : ""}
              </Text>
              {subline ? (
                <Text variant="small" tone="muted" className="truncate">
                  {subline}
                </Text>
              ) : null}
            </div>
          </div>
          <div className="flex shrink-0 items-center gap-2">
            {tag ? (
              <Badge
                variant={tag.tone === "done" ? "success" : "info"}
                className={TAG_CLASS[tag.tone]}
              >
                {tag.label}
              </Badge>
            ) : null}
            <Button
              as="NextLink"
              href={`/copilot?sessionId=${sentFrom.sessionId}`}
              variant="secondary"
              size="xs"
              leftIcon={<Icon icon={ArrowLeft01Icon} size={14} />}
            >
              Back to {from}&apos;s thread
            </Button>
          </div>
        </div>
        <div className="flex flex-col gap-4 sm:flex-row sm:gap-6">
          <div className="flex min-w-0 flex-1 flex-col gap-1">
            <Text variant="eyebrow">Brief</Text>
            <Text variant="body" tone="primary" className="line-clamp-4">
              {brief || "No brief was sent with this hand-off."}
            </Text>
          </div>
          <ExpectedBack
            from={from}
            capLine={getCapLine(capUsd, delegation?.cost_usd)}
          />
        </div>
      </section>
    </div>
  );
}
