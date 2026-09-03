import {
  AlertDiamondIcon,
  ArrowUpRight01Icon,
  CheckListIcon,
} from "@hugeicons/core-free-icons";
import Link from "next/link";
import type { HomeBriefingOutcome } from "@/app/api/__generated__/models/homeBriefingOutcome";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { ExpertAvatar } from "@/components/molecules/ExpertAvatar/ExpertAvatar";
import { cn } from "@/lib/utils";
import { formatCurrency, formatDuration } from "../../../helpers";

interface Props {
  outcome: HomeBriefingOutcome;
}

const ROW_CLASS = "flex gap-3 px-4 py-3";

export function OutcomeRow({ outcome }: Props) {
  const failed = outcome.status === "failed";
  const content = (
    <>
      <span
        className={cn(
          "mt-0.5 flex size-6 shrink-0 items-center justify-center rounded-md",
          failed ? "bg-rose-50 text-rose-600" : "bg-zinc-100 text-zinc-500",
        )}
      >
        <Icon
          icon={failed ? AlertDiamondIcon : CheckListIcon}
          size={13}
          aria-hidden="true"
        />
      </span>
      <div className="min-w-0 flex-1">
        <Text
          variant="body-medium"
          className="text-pretty text-[13px] leading-5 text-zinc-900"
        >
          {outcome.title}
        </Text>
        <Text
          variant="small"
          className="mt-0.5 line-clamp-2 text-pretty text-[13px] leading-5 text-zinc-500"
        >
          {outcome.summary}
        </Text>
        <div className="mt-1 flex flex-wrap items-center gap-x-1.5 gap-y-1 text-[11px] text-zinc-400">
          {outcome.expert ? (
            <ExpertAvatar
              name={outcome.expert.name}
              avatarUrl={outcome.expert.avatar_url}
              size={16}
            />
          ) : null}
          <span className="font-medium text-zinc-500">
            {outcome.expert?.name ?? outcome.agent_name}
          </span>
          <span aria-hidden="true">·</span>
          <span className="tabular-nums">
            {formatOccurredAt(outcome.occurred_at)}
          </span>
          {outcome.duration_seconds ? (
            <>
              <span aria-hidden="true">·</span>
              <span className="tabular-nums">
                {formatDuration(outcome.duration_seconds)}
              </span>
            </>
          ) : null}
          {outcome.cost_cents ? (
            <>
              <span aria-hidden="true">·</span>
              <span className="tabular-nums">
                {formatCurrency(outcome.cost_cents)}
              </span>
            </>
          ) : null}
        </div>
      </div>
      {outcome.link ? (
        <Icon
          icon={ArrowUpRight01Icon}
          size={14}
          className="mt-1 shrink-0 text-zinc-300 transition-colors group-hover:text-zinc-600"
          aria-hidden="true"
        />
      ) : null}
    </>
  );

  if (!outcome.link) {
    return <article className={ROW_CLASS}>{content}</article>;
  }
  return (
    <Link
      href={outcome.link}
      className={cn(
        ROW_CLASS,
        "group outline-none transition-colors hover:bg-zinc-50 focus-visible:bg-zinc-50",
      )}
    >
      {content}
    </Link>
  );
}

function formatOccurredAt(value: Date | null | undefined) {
  if (!value) return "Recently";
  return new Intl.DateTimeFormat(undefined, {
    hour: "numeric",
    minute: "2-digit",
  }).format(new Date(value));
}
