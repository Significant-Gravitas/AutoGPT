"use client";

import { useOptionalSidebar } from "@/components/molecules/Sidebar/Sidebar";
import { useRouter } from "next/navigation";
import { useFirstRunTourStore } from "./store";

// The walkthrough ends on the first tasks, which live on the empty home page.
export function useShowMeAround() {
  const router = useRouter();
  const open = useFirstRunTourStore((state) => state.open);
  const requestReplay = useFirstRunTourStore((state) => state.requestReplay);
  const sidebar = useOptionalSidebar();

  return function showMeAround() {
    // The phone's sheet is modal and would trap the walkthrough's focus.
    if (sidebar?.isMobile) sidebar.setOpenMobile(false);
    const { pathname, search } = window.location;
    if (pathname === "/home" && search === "") {
      open("menu");
      return;
    }
    // From another page it opens once Home has loaded (useReplayOnArrival),
    // not over the page being left and that page's sidebar state.
    requestReplay();
    router.push("/home");
  };
}
