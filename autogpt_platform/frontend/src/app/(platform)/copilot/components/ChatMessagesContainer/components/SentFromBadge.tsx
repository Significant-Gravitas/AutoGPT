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
      // TODO(tokens): no exact role for bg-purple-100, text-purple-800, hover:bg-purple-200
      className="inline-flex items-center gap-1 rounded-full bg-purple-100 px-2 py-0.5 text-caption-sm font-medium leading-inherit text-purple-800 transition-colors hover:bg-purple-200"
    >
      <Icon icon={ArrowTurnBackwardIcon} size={12} />
      Sent from {name}
    </Link>
  );
}
