"use client";

import { cn } from "@/lib/utils";
import { Clock01Icon, PinIcon } from "@hugeicons/core-free-icons";
import { HomeRecentChats } from "./HomeRecentChats";
import { useHomeRecentChats } from "./useHomeRecentChats";
import { usePinChat } from "./usePinChat";

interface Props {
  /** Whose recent chats to list; null is Otto. */
  expertId: string | null;
}

/** Under the composer: the recipient's recent chats, at the composer's
 *  width, and beside them the chats the user pinned, once there are any;
 *  nothing when there are no chats. The box's gray is a thin wash, so it
 *  is laid over the page's own colour and Home's dither trail stays behind
 *  it. */
export function HomeRecentRow({ expertId }: Props) {
  const { rows, hasMore, isLoadingMore, loadMore } =
    useHomeRecentChats(expertId);
  const pin = usePinChat();
  if (rows.length === 0) return null;

  const pinned = rows.filter((row) => row.session.is_pinned);
  const recent = rows.filter((row) => !row.session.is_pinned);
  const hasBoth = pinned.length > 0 && (recent.length > 0 || hasMore);

  return (
    <div
      data-testid="home-tray-row"
      className={cn(
        "mx-auto mt-20 mb-6 grid w-full gap-3",
        hasBoth ? "max-w-[64rem] md:grid-cols-2" : "max-w-[42rem]",
      )}
    >
      {recent.length > 0 || hasMore ? (
        <div className="min-w-0 rounded-composer-tray bg-canvas">
          <HomeRecentChats
            title="Recent chats"
            icon={Clock01Icon}
            headingID="home-recent-chats"
            rows={recent}
            hasMore={hasMore}
            isLoadingMore={isLoadingMore}
            onLoadMore={loadMore}
            onTogglePin={pin.togglePin}
            pendingPinID={pin.pendingID}
          />
        </div>
      ) : null}
      {pinned.length > 0 ? (
        <div className="min-w-0 rounded-composer-tray bg-canvas">
          <HomeRecentChats
            title="Pinned chats"
            icon={PinIcon}
            headingID="home-pinned-chats"
            rows={pinned}
            hasMore={false}
            isLoadingMore={false}
            onLoadMore={loadMore}
            onTogglePin={pin.togglePin}
            pendingPinID={pin.pendingID}
          />
        </div>
      ) : null}
    </div>
  );
}
