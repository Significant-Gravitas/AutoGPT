import { useOptionalSidebar } from "@/components/ui/sidebar";
import { useEffect, useRef } from "react";

export function useSoulPanelSidebarCollapse(isPanelOpen: boolean) {
  const sidebar = useOptionalSidebar();
  const openBeforePanelRef = useRef<boolean | null>(null);
  const openRef = useRef(sidebar?.open);
  openRef.current = sidebar?.open;
  const wasPanelOpenRef = useRef(false);

  useEffect(() => {
    if (!sidebar || isPanelOpen === wasPanelOpenRef.current) return;
    wasPanelOpenRef.current = isPanelOpen;

    if (isPanelOpen) {
      openBeforePanelRef.current = openRef.current ?? false;
      sidebar.setOpen(false);
      sidebar.setOpenMobile(false);
      return;
    }

    if (openBeforePanelRef.current !== null && openRef.current === false) {
      sidebar.setOpen(openBeforePanelRef.current);
    }
    openBeforePanelRef.current = null;
  }, [isPanelOpen, sidebar]);
}
