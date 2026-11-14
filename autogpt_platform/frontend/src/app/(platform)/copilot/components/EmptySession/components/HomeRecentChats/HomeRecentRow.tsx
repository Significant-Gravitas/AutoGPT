"use client";

import { useSessionList } from "@/app/(platform)/copilot/useSessionList";
import { HomeRecentChats } from "./HomeRecentChats";
import { recentChatsFor } from "./helpers";

interface Props {
  /** Whose recent chats to list; null is Otto. */
  expertId: string | null;
}

/** Under the composer: the recipient's recent chats, at the composer's
 *  width; nothing when there are none. The box's gray is a thin wash, so it
 *  is laid over the page's own colour and Home's dither trail stays behind
 *  it. */
export function HomeRecentRow({ expertId }: Props) {
  const { sessions } = useSessionList();
  const recent = recentChatsFor(sessions, expertId);
  if (recent.length === 0) return null;

  return (
    <div
      data-testid="home-tray-row"
      className="mx-auto mt-20 mb-6 w-full max-w-[42rem] rounded-composer-tray bg-canvas"
    >
      <HomeRecentChats recent={recent} />
    </div>
  );
}
