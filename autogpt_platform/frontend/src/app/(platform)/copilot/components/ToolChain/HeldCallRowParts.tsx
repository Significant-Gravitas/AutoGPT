import { Tick02Icon } from "@hugeicons/core-free-icons";
import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { AUTOPILOT_NAME } from "@/components/molecules/AutopilotAvatar/helpers";
import { cn } from "@/lib/utils";
import { approvalCardId } from "../ApprovalQueue/helpers";
import type { HeldRowInfo, HeldState } from "./heldRow";

const TAGS: Record<HeldState, { text: string; className: string }> = {
  waiting: { text: "Waiting for you", className: "bg-amber-50 text-amber-700" },
  approved: { text: "Approved", className: "bg-zinc-100 text-zinc-500" },
  rejected: { text: "Rejected", className: "bg-zinc-100 text-zinc-500" },
  expired: { text: "Expired", className: "bg-zinc-100 text-zinc-500" },
  closed: { text: "Not run", className: "bg-zinc-100 text-zinc-500" },
  unknown: { text: "Unclear", className: "bg-amber-50 text-amber-700" },
  "not-run": { text: "Not run", className: "bg-red-50 text-red-700" },
};

// A held read is released or kept out, not approved or rejected.
const READ_TAGS: Partial<Record<HeldState, string>> = {
  waiting: "Held",
  approved: "Released",
  rejected: "Kept out",
};

export function HeldTag({ state, read }: { state: HeldState; read?: boolean }) {
  const tag = TAGS[state];
  const text = (read && READ_TAGS[state]) || tag.text;
  return (
    <span
      className={cn(
        "inline-flex shrink-0 items-center gap-1 rounded-md px-1.5 py-px text-xs font-medium",
        tag.className,
      )}
    >
      {state === "approved" && <Icon icon={Tick02Icon} size={11} aria-hidden />}
      {text}
    </span>
  );
}

const DETAIL: Record<Exclude<HeldState, "approved">, string> = {
  waiting: `Nothing has run yet. ${AUTOPILOT_NAME} carried on without it.`,
  rejected: `You rejected this, so it didn't run. ${AUTOPILOT_NAME} was told.`,
  expired: `Approved, but it didn't run within an hour, so the approval lapsed. Ask ${AUTOPILOT_NAME} to try again.`,
  unknown: `It may have run, but its result was lost. Check before relying on it.`,
  closed: `It couldn't be carried out, so nothing ran. Ask ${AUTOPILOT_NAME} to try again if it's still needed.`,
  "not-run": `${AUTOPILOT_NAME} couldn't open an approval for this, so nothing ran.`,
};

const READ_DETAIL: Partial<Record<HeldState, string>> = {
  waiting: `Nothing from it has reached ${AUTOPILOT_NAME} yet. ${AUTOPILOT_NAME} carried on without it.`,
  rejected: `You kept this out, so ${AUTOPILOT_NAME} never saw it.`,
};

export function HeldCallDetail({ held }: { held: HeldRowInfo }) {
  if (held.state === "approved") return null;
  const reviewId = held.reviewId;
  function goToApproval() {
    if (!reviewId) return;
    const el = document.getElementById(approvalCardId(reviewId));
    el?.scrollIntoView?.({ block: "center", behavior: "smooth" });
    el?.focus({ preventScroll: true });
  }
  return (
    <div className="flex flex-col items-start gap-1 rounded-lg border border-zinc-200 px-3 py-2 text-sm text-zinc-600">
      <p>{(held.read && READ_DETAIL[held.state]) || DETAIL[held.state]}</p>
      {held.state === "waiting" && reviewId && (
        <Button
          variant="link"
          size="small"
          className="h-auto min-w-0 px-0 py-0 text-sm"
          onClick={goToApproval}
        >
          Go to the approval
        </Button>
      )}
    </div>
  );
}
