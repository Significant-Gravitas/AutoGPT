"use client";

import { usePathname, useSearchParams } from "next/navigation";
import { useCopilotNotifications } from "../../useCopilotNotifications";

// Mounted on every platform page, not only the chat page, so a chat that
// finishes while the user is in the library still notifies and still shows
// as finished in the sidebar.
export function CopilotNotificationsListener() {
  const pathname = usePathname();
  const searchParams = useSearchParams();
  const activeSessionID =
    pathname === "/home" ? searchParams.get("sessionId") : null;

  useCopilotNotifications(activeSessionID);
  return null;
}
