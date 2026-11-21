import { Icon } from "@/components/atoms/Icon/Icon";
import { ArrowTurnBackwardIcon } from "@hugeicons/core-free-icons";
import Link from "next/link";
import { getSentFromDisplayName, type SentFrom } from "../../../sentFrom";
import { useExpertMap } from "../../../useExpertMap";

interface Props {
  sentFrom: SentFrom;
}

export function SentFromBadge({ sentFrom }: Props) {
  const { expertsById } = useExpertMap();
  const resolvedName = sentFrom.expertId
    ? expertsById.get(sentFrom.expertId)?.name
    : null;
  const name = getSentFromDisplayName(sentFrom, resolvedName);

  return (
    <Link
      href={`/copilot?sessionId=${sentFrom.sessionId}`}
      data-testid="sent-from-badge"
      title="Open the thread this task was sent from"
      className="inline-flex items-center gap-1 rounded-full bg-accent-surface px-2 py-0.5 text-caption-sm leading-inherit font-medium text-accent-strong transition-colors hover:bg-brand/50"
    >
      <Icon icon={ArrowTurnBackwardIcon} size={12} />
      Sent from {name}
    </Link>
  );
}
