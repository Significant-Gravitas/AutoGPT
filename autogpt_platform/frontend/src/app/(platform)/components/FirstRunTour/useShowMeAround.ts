"use client";

import { useOptionalSidebar } from "@/components/molecules/Sidebar/Sidebar";
import { useRouter } from "next/navigation";
import { useFirstRunTourStore } from "./store";

// The walkthrough ends on the first tasks, which live on the empty home page.
export function useShowMeAround() {
  const router = useRouter();
  const open = useFirstRunTourStore((state) => state.open);
  const sidebar = useOptionalSidebar();

  return function showMeAround() {
    // The phone's sheet is modal and would trap the walkthrough's focus.
    if (sidebar?.isMobile) sidebar.setOpenMobile(false);
    const { pathname, search } = window.location;
    if (pathname !== "/home" || search !== "") router.push("/home");
    open("menu");
  };
}
