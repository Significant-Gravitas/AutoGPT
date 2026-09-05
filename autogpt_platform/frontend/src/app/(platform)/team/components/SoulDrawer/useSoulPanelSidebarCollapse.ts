import { useOptionalSidebar } from "@/components/ui/sidebar";
import { useEffect, useLayoutEffect, useRef } from "react";

export function useSoulPanelSidebarCollapse(isPanelOpen: boolean) {
  const sidebar = useOptionalSidebar();
  const sidebarRef = useRef(sidebar);
  useLayoutEffect(() => {
    sidebarRef.current = sidebar;
  }, [sidebar]);
  const hasSidebar = Boolean(sidebar);

  useEffect(() => {
    const current = sidebarRef.current;
    if (!isPanelOpen || !current) return;
    const wasOpen = current.open;
    current.setOpen(false);
    current.setOpenMobile(false);

    return () => {
      const latest = sidebarRef.current;
      if (wasOpen && latest && !latest.open) latest.setOpen(true);
    };
  }, [isPanelOpen, hasSidebar]);
}
