"use client";

import { usePathname } from "next/navigation";
import { ChatSessionList } from "./ChatSessionList";

export function SidebarDynamicContent() {
  const pathname = usePathname();

  if (pathname.startsWith("/copilot")) {
    return <ChatSessionList />;
  }

  return null;
}
