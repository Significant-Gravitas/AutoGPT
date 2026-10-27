"use client";

import {
  Alert02Icon,
  Clock01Icon,
  InformationCircleIcon,
} from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";
import { ExpertAvatar } from "@/components/molecules/ExpertAvatar/ExpertAvatar";
import { cn } from "@/lib/utils";
import type { HandoffState } from "./expertHandoff";

const BADGES: Record<
  Exclude<HandoffState, "unconfirmed" | "unknown">,
  { className: string; icon: typeof Clock01Icon | null }
> = {
  working: { className: "bg-info-surface text-info", icon: null },
  waiting: {
    className: "bg-warning-surface text-warning-foreground",
    icon: Clock01Icon,
  },
  // Neutral, not success: an idle thread may hold a question or an error.
  stopped: {
    className: "bg-surface-hover text-foreground-secondary",
    icon: InformationCircleIcon,
  },
  failed: {
    className: "bg-warning-surface text-warning-foreground",
    icon: Alert02Icon,
  },
};

interface AvatarProps {
  name: string;
  avatarUrl: string | null;
  color: string | null;
  state: HandoffState | null;
  size: number;
}

/** The expert with what their thread is doing on the corner. The badge is
 *  decoration: the sentence beside it always says the same in words. */
export function StatusAvatar({
  name,
  avatarUrl,
  color,
  state,
  size,
}: AvatarProps) {
  const badge =
    state && state !== "unconfirmed" && state !== "unknown"
      ? BADGES[state]
      : null;
  const badgeSize = Math.round(size * 0.5);
  return (
    <span className="relative inline-flex shrink-0">
      <ExpertAvatar
        name={name}
        avatarUrl={avatarUrl}
        color={color}
        size={size}
      />
      {badge && (
        <span
          aria-hidden
          data-testid="handoff-badge"
          style={{ width: badgeSize, height: badgeSize }}
          className={cn(
            "absolute -bottom-0.5 -end-0.5 flex items-center justify-center rounded-full ring-2 ring-surface",
            badge.className,
          )}
        >
          {badge.icon ? (
            <Icon icon={badge.icon} size={Math.round(badgeSize * 0.7)} />
          ) : (
            <span className="size-[60%] animate-spin rounded-full border-[1.5px] border-current border-t-transparent motion-reduce:animate-none" />
          )}
        </span>
      )}
    </span>
  );
}
