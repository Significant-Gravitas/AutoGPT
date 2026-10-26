import {
  Cancel01Icon,
  InformationCircleIcon,
  Tick02Icon,
} from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { AUTOPILOT_NAME } from "@/components/molecules/AutopilotAvatar/helpers";
import { HeadlineText } from "@/app/(platform)/copilot/components/ApprovalQueue/components/ApprovalHeadline";
import { toApprovalItem } from "@/app/(platform)/copilot/components/ApprovalQueue/helpers";
import { cn } from "@/lib/utils";
import type { useReviewCarousel } from "../useReviewCarousel";
import { HeldAvatar } from "./HeldAvatar";

interface Props {
  carousel: ReturnType<typeof useReviewCarousel>;
  // Below lg the list is a sheet over the card.
  sheet?: boolean;
}

const OUTCOME_ICONS = {
  approved: [Tick02Icon, "text-success"],
  rejected: [Cancel01Icon, "text-foreground-subtle"],
  elsewhere: [InformationCircleIcon, "text-foreground-subtle"],
} as const;

export function ReviewSidebar({ carousel, sheet = false }: Props) {
  return (
    <nav
      aria-label={sheet ? "All held calls" : "Held calls"}
      className="flex min-h-0 w-full flex-col"
    >
      <div className="flex items-baseline gap-2 px-4 pb-2 pt-4">
        <Text variant="body-medium" as="h2">
          To review
        </Text>
        <Text
          variant="small"
          as="span"
          tone="secondary"
          className="tabular-nums"
        >
          {carousel.left} of {carousel.total} left
        </Text>
      </div>
      <ul className="min-h-0 flex-1 overflow-y-auto overscroll-contain pb-2">
        {carousel.order.map(({ item, receipt }) => {
          const isCurrent = carousel.current?.item.id === item.id;
          const outcome = receipt ? OUTCOME_ICONS[receipt.outcome] : null;
          return (
            <li key={item.id}>
              <button
                type="button"
                aria-current={isCurrent ? "true" : undefined}
                onClick={() => carousel.jump(item.id)}
                className={cn(
                  "flex w-full items-start gap-2.5 border-l-2 py-2 pl-3.5 pr-4 text-left text-sm hover:bg-surface focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-border-strong",
                  isCurrent
                    ? "border-action-hover bg-surface"
                    : "border-transparent",
                  receipt
                    ? "text-foreground-subtle"
                    : "text-foreground-emphasis",
                )}
              >
                <HeldAvatar item={item} size={20} />
                <span className="flex min-w-0 flex-1 flex-col">
                  <span className="text-xs text-foreground-tertiary">
                    {item.expert?.name ?? AUTOPILOT_NAME}
                  </span>
                  <span className="flex min-w-0 items-center gap-1.5">
                    {outcome ? (
                      <Icon
                        icon={outcome[0]}
                        size={14}
                        className={cn("shrink-0", outcome[1])}
                        aria-hidden
                      />
                    ) : null}
                    <span className="min-w-0 truncate">
                      <HeadlineText item={toApprovalItem(item.review!)} />
                      {receipt ? (
                        <span className="sr-only">, {receipt.text}</span>
                      ) : null}
                    </span>
                  </span>
                </span>
                {carousel.isNew(item.id) ? (
                  <span
                    // TODO(tokens): no fitting role for text-blue-700 #1D4ED8; #1D4ED8 is sidebar-primary, a dark-theme sidebar fill (light #18181B)
                    className="shrink-0 rounded bg-info-surface px-1.5 text-xs font-medium text-blue-700"
                  >
                    new
                  </span>
                ) : null}
              </button>
            </li>
          );
        })}
      </ul>
      <Text
        variant="small"
        tone="secondary"
        className="border-t border-separator px-4 py-3"
      >
        {sheet ? "Tap an item to jump to it" : "← → move · Esc close"}
      </Text>
    </nav>
  );
}
