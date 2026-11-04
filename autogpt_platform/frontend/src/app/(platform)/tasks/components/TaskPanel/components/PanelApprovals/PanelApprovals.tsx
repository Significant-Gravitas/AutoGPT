import type { CardApproval } from "@/app/api/__generated__/models/cardApproval";
import { Text } from "@/components/atoms/Text/Text";
import { CardApprovals } from "../../../CardApprovals/CardApprovals";

interface Props {
  taskId: string;
  askerName: string;
  cards: CardApproval[];
}

export function PanelApprovals({ taskId, askerName, cards }: Props) {
  return (
    <section
      aria-label="Needs you"
      className="flex animate-inbox-enter flex-col gap-3 rounded-card bg-surface-wash p-4"
    >
      <div className="flex items-center gap-2">
        <span
          aria-hidden
          className="size-2 shrink-0 rounded-full bg-status-waiting"
        />
        <Text variant="inbox-medium" tone="waiting">
          Needs you
        </Text>
        <Text variant="inbox" tone="quiet">
          · {askerName} needs your OK
        </Text>
      </div>
      <CardApprovals taskId={taskId} cards={cards} />
    </section>
  );
}
