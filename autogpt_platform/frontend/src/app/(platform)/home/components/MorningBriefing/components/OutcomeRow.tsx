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
import { trackFunnel } from "@/services/experts/experts-analytics";
import { cn } from "@/lib/utils";
import { formatCurrency, formatDuration } from "../../../helpers";

interface Props {
  outcome: HomeBriefingOutcome;
}

export function OutcomeRow({ outcome }: Props) {
  const failed = outcome.status === "failed";
  const content = (
    <>
      <span
        className={cn(
          "mt-0.5 flex size-8 shrink-0 items-center justify-center rounded-lg",
          failed ? "bg-rose-50 text-rose-600" : "bg-zinc-100 text-zinc-500",
        )}
      >
        <Icon
          icon={failed ? AlertDiamondIcon : CheckListIcon}
          size={16}
          aria-hidden="true"
        />
      </span>
      <div className="min-w-0 flex-1">
        <Text variant="body-medium" className="text-pretty text-zinc-950">
          {outcome.title}
        </Text>
        <Text
          variant="body"
          className="mt-1 line-clamp-2 text-pretty text-zinc-600"
        >
          {outcome.summary}
        </Text>
        <div className="mt-1.5 flex flex-wrap items-center gap-x-2 gap-y-1 text-xs text-zinc-400">
          {outcome.expert ? (
            <ExpertAvatar
              name={outcome.expert.name}
              avatarUrl={outcome.expert.avatar_url}
              size={22}
            />
          ) : null}
          <span>{outcome.expert?.name ?? outcome.agent_name}</span>
          <span aria-hidden="true">·</span>
          <span>{formatOccurredAt(outcome.occurred_at)}</span>
          {outcome.duration_seconds ? (
            <>
              <span aria-hidden="true">·</span>
              <span>{formatDuration(outcome.duration_seconds)}</span>
            </>
          ) : null}
          {outcome.cost_cents ? (
            <>
              <span aria-hidden="true">·</span>
              <span>{formatCurrency(outcome.cost_cents)}</span>
            </>
          ) : null}
        </div>
      </div>
      {outcome.link ? (
        <Icon
          icon={ArrowUpRight01Icon}
          size={17}
          className="mt-1 shrink-0 text-zinc-300 transition-colors group-hover:text-zinc-700"
          aria-hidden="true"
        />
      ) : null}
    </>
  );

  if (!outcome.link) {
    return (
      <article className="flex gap-3 px-4 py-3 sm:px-5">{content}</article>
    );
  }
  return (
    <Link
      href={outcome.link}
      onClick={() =>
        trackFunnel("briefing_outcome_clicked", { status: outcome.status })
      }
      className="group flex gap-3 px-4 py-3 outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-zinc-400 sm:px-5"
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
