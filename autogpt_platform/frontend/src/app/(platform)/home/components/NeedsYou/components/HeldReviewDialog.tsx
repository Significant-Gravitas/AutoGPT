import {
  ArrowLeft01Icon,
  ArrowRight01Icon,
  Menu01Icon,
  Tick02Icon,
} from "@hugeicons/core-free-icons";
import Link from "next/link";
import type { KeyboardEvent, ReactNode } from "react";
import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import {
  ApprovalActions,
  ApprovalCard,
} from "@/app/(platform)/copilot/components/ApprovalQueue/components/ApprovalCard/ApprovalCard";
import {
  modeLine,
  toApprovalItem,
} from "@/app/(platform)/copilot/components/ApprovalQueue/helpers";
import { isKey } from "@/lib/keyboard";
import { reviewTally, shortAge } from "../helpers";
import type { useHeldReview } from "../useHeldReview";
import type { useReviewCarousel } from "../useReviewCarousel";
import { HeldAvatar } from "./HeldAvatar";
import { HeldReceiptRow } from "./HeldReceiptRow";
import { ReviewSidebar } from "./ReviewSidebar";

interface Props {
  carousel: ReturnType<typeof useReviewCarousel>;
  held: ReturnType<typeof useHeldReview>;
}

// Every held call on the page, one at a time: the chat's card, a sidebar, and ‹ › to move.
export function HeldReviewDialog({ carousel, held }: Props) {
  const { current } = carousel;
  const approval = current ? toApprovalItem(current.item.review!) : null;

  function handleKeyDown(e: KeyboardEvent<HTMLDivElement>) {
    // The rule menu portals out of this subtree; its arrows are the menu's.
    if (!e.currentTarget.contains(e.target as Node) || e.defaultPrevented)
      return;
    if (isKey(e, "ArrowLeft")) carousel.move(-1);
    else if (isKey(e, "ArrowRight")) carousel.move(1);
  }

  return (
    <Dialog
      controlled={{
        isOpen: carousel.open,
        set: (isOpen) => (isOpen ? undefined : carousel.close()),
      }}
      className="h-[90vh] p-0 lg:h-[min(720px,85vh)] lg:w-[960px] lg:min-w-0 lg:max-w-[95vw]"
    >
      <Dialog.Content>
        <div
          onKeyDown={handleKeyDown}
          className="relative -mx-2 flex h-[calc(90vh-1rem)] min-h-0 lg:h-[min(720px,85vh)]"
        >
          <div className="hidden w-[272px] shrink-0 border-r border-zinc-100 bg-zinc-50 lg:flex">
            <ReviewSidebar carousel={carousel} />
          </div>
          <section
            aria-label="Held call"
            className="flex min-w-0 flex-1 flex-col"
          >
            <header className="flex min-h-16 shrink-0 items-center gap-2 border-b border-zinc-100 py-3 pl-4 pr-16">
              <Button
                variant="secondary"
                size="icon-sm"
                className="lg:hidden"
                aria-label="All held calls"
                aria-expanded={carousel.sheetOpen}
                leadingIcon={Menu01Icon}
                onClick={() => carousel.setSheetOpen(!carousel.sheetOpen)}
              />
              <Button
                variant="secondary"
                size="icon-sm"
                aria-label="Previous"
                disabled={!carousel.canPrev}
                leadingIcon={ArrowLeft01Icon}
                onClick={() => carousel.move(-1)}
              />
              <Text
                variant="body-medium"
                as="span"
                aria-live="polite"
                className="tabular-nums"
              >
                {carousel.position} of {carousel.total}
              </Text>
              <Button
                variant="secondary"
                size="icon-sm"
                aria-label="Next"
                disabled={!carousel.canNext}
                leadingIcon={ArrowRight01Icon}
                onClick={() => carousel.move(1)}
              />
              {current && !carousel.finished ? (
                <span className="ml-2 flex min-w-0 items-center gap-2">
                  <HeldAvatar item={current.item} size={24} />
                  <Text
                    variant="body"
                    as="span"
                    tone="secondary"
                    className="truncate"
                  >
                    {current.item.expert?.name ?? "Otto"}
                    {current.item.created_at
                      ? ` · ${shortAge(current.item.created_at)}`
                      : ""}
                  </Text>
                </span>
              ) : null}
            </header>
            {/* Only the body scrolls; the header and the decision footer stay put. */}
            <div
              ref={carousel.paneRef}
              data-testid="review-pane-body"
              className="min-h-0 flex-1 overflow-y-auto overscroll-contain"
            >
              {carousel.finished ? (
                <Finished carousel={carousel} />
              ) : current?.receipt && approval ? (
                <div className="px-2 py-4">
                  <HeldReceiptRow
                    approval={approval}
                    receipt={current.receipt}
                    heading
                  />
                </div>
              ) : current && approval ? (
                <ApprovalCard
                  key={current.item.id}
                  item={approval}
                  status={held.statusOf(current.item.id)}
                  failed={held.hasFailed(current.item.id)}
                  expertName={current.item.expert?.name ?? null}
                  note={
                    approval.reasonKind === "mode"
                      ? modeLine(approval.mode)
                      : null
                  }
                  onApprove={(rule, scope) =>
                    carousel.decide(current.item, true, rule, scope)
                  }
                  onReject={() => carousel.decide(current.item, false)}
                  hideActions
                />
              ) : null}
            </div>
            {carousel.finished ? (
              <PaneFooter>
                <Button size="small" variant="primary" onClick={carousel.close}>
                  Done
                </Button>
              </PaneFooter>
            ) : current && approval && !current.receipt ? (
              <PaneFooter>
                <ApprovalActions
                  key={current.item.id}
                  item={approval}
                  status={held.statusOf(current.item.id)}
                  expertName={current.item.expert?.name ?? null}
                  onApprove={(rule, scope) =>
                    carousel.decide(current.item, true, rule, scope)
                  }
                  onReject={() => carousel.decide(current.item, false)}
                  aside={
                    <Link
                      href={current.item.primary_action.href}
                      className="rounded-sm text-sm font-medium text-zinc-800 underline underline-offset-4 hover:text-zinc-950 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-zinc-300"
                    >
                      Open chat
                    </Link>
                  }
                />
              </PaneFooter>
            ) : null}
          </section>
          {carousel.sheetOpen ? (
            <div className="absolute inset-x-0 bottom-0 top-16 z-10 flex rounded-t-2xl border-t border-zinc-200 bg-white shadow-lg lg:hidden">
              <ReviewSidebar carousel={carousel} sheet />
            </div>
          ) : null}
        </div>
      </Dialog.Content>
    </Dialog>
  );
}

function Finished({
  carousel,
}: {
  carousel: ReturnType<typeof useReviewCarousel>;
}) {
  return (
    <div className="flex h-full flex-col items-center justify-center gap-2 px-6 text-center">
      <Icon
        icon={Tick02Icon}
        size={20}
        className="text-green-600"
        aria-hidden
      />
      <Text variant="h4" as="h3" data-pane-focus>
        All {carousel.total} reviewed
      </Text>
      <Text variant="body" tone="secondary">
        {reviewTally(carousel.order)}
      </Text>
    </div>
  );
}

function PaneFooter({ children }: { children: ReactNode }) {
  return (
    <footer
      data-testid="review-pane-footer"
      className="flex shrink-0 justify-end border-t border-zinc-100 bg-white px-4 py-3 [&>div]:w-full"
    >
      {children}
    </footer>
  );
}
