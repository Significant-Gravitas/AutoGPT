"use client";

import type { CardApproval } from "@/app/api/__generated__/models/cardApproval";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { approveLabel, cardLine } from "./helpers";
import { useCardApprovals } from "./useCardApprovals";

interface Props {
  taskId: string;
  cards: CardApproval[];
}

/** The hire and raise cards a task waits on, answered right here: the same
 *  Approve the card in the task's chat sends. */
export function CardApprovals({ taskId, cards }: Props) {
  const { approve, decline, pending } = useCardApprovals(taskId, cards);

  return (
    <div className="flex flex-col gap-2" data-testid="card-approvals">
      <ul className="flex flex-col gap-1.5">
        {cards.map((card) => (
          <li key={card.confirmation_id} className="flex min-w-0 flex-col">
            <Text variant="inbox-medium" tone="ink" className="truncate">
              {cardLine(card)}
            </Text>
            {card.tagline ? (
              <Text variant="inbox" tone="quiet" className="line-clamp-2">
                {card.tagline}
              </Text>
            ) : null}
          </li>
        ))}
      </ul>
      <div className="flex flex-wrap items-center gap-1.5">
        <Button
          variant="primary"
          size="xs"
          loading={pending === "approve"}
          disabled={pending !== null}
          onClick={approve}
        >
          {approveLabel(cards.length)}
        </Button>
        <Button
          variant="ghost-muted"
          size="xs"
          loading={pending === "decline"}
          disabled={pending !== null}
          onClick={decline}
        >
          Decline
        </Button>
      </div>
    </div>
  );
}
