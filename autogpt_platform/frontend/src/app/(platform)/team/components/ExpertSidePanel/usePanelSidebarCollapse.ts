import {
  getPersistedSidebarOpen,
  useOptionalSidebar,
} from "@/components/ui/sidebar";
import { useEffect, useLayoutEffect, useRef } from "react";

export function usePanelSidebarCollapse(isPanelOpen: boolean) {
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
    current.setOpen(false, { persist: false });
    current.setOpenMobile(false);

    return () => {
      const latest = sidebarRef.current;
      // A toggle made while the panel was up is remembered, and wins.
      if (wasOpen && latest && !latest.open)
        latest.setOpen(getPersistedSidebarOpen() ?? true, { persist: false });
    };
  }, [isPanelOpen, hasSidebar]);
}
