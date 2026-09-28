import { ArrowRight01Icon } from "@hugeicons/core-free-icons";
import Link from "next/link";
import type { DelegationSummary } from "@/app/api/__generated__/models/delegationSummary";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";
import { ExpertAvatar } from "@/components/molecules/ExpertAvatar/ExpertAvatar";
import { DelegationStatusBadge } from "./DelegationStatusBadge";
import { formatDelegationTime, getDelegationHref } from "./helpers";

interface Props {
  delegation: DelegationSummary;
  meta: string;
}

export function DelegationListRow({ delegation, meta }: Props) {
  const href = getDelegationHref(delegation);
  const expert = delegation.expert;
  const content = (
    <>
      <ExpertAvatar
        name={expert?.name ?? null}
        avatarUrl={expert?.avatar_url ?? null}
        color={expert?.color ?? null}
        size={36}
        className="shrink-0"
      />
      <div className="flex min-w-0 flex-1 flex-col gap-0.5">
        <Text variant="body-medium" tone="primary" className="truncate">
          {delegation.title}
        </Text>
        {delegation.brief ? (
          <Text variant="body" tone="secondary" className="truncate">
            {delegation.brief}
          </Text>
        ) : null}
        <Text variant="small" tone="muted" className="truncate">
          {meta}
        </Text>
      </div>
      <div className="flex shrink-0 items-center gap-4">
        <Text
          variant="small"
          as="span"
          tone="muted"
          className="hidden w-[4.5rem] text-right tabular-nums sm:block"
        >
          {formatDelegationTime(delegation.created_at)}
        </Text>
        <span className="flex justify-end sm:w-28">
          <DelegationStatusBadge status={delegation.status} />
        </span>
        <Icon
          icon={ArrowRight01Icon}
          size={16}
          className="text-zinc-900"
          aria-hidden="true"
        />
      </div>
    </>
  );
  const rowClass =
    "flex items-center gap-4 border-b border-zinc-100 px-1 py-3.5 outline-none";

  if (!href) return <div className={rowClass}>{content}</div>;
  return (
    <Link
      href={href}
      className={cn(
        rowClass,
        "rounded-sm transition-colors hover:bg-zinc-50 focus-visible:bg-zinc-50",
      )}
    >
      {content}
    </Link>
  );
}
