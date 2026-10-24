"use client";

import { useOptionalSidebar } from "@/components/ui/sidebar";
import { useRouter } from "next/navigation";
import { useFirstRunTourStore } from "./store";

// "Show me around" in the profile menu: replays the walkthrough from the
// start. It ends on the first tasks, which live on the empty home page, so
// it goes there first from anywhere else (an open chat included).
export function useShowMeAround() {
  const router = useRouter();
  const open = useFirstRunTourStore((state) => state.open);
  const sidebar = useOptionalSidebar();

  return function showMeAround() {
    // On a phone the menu lives in the sidebar sheet, a modal drawer that
    // would sit over the walkthrough and trap its focus.
    if (sidebar?.isMobile) sidebar.setOpenMobile(false);
    const { pathname, search } = window.location;
    if (pathname !== "/home" || search !== "") router.push("/home");
    open("menu");
  };
}
