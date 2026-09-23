import { useCopilotUIStore } from "@/app/(platform)/copilot/store";
import { useSidebar } from "@/components/ui/sidebar";
import { matchesRoute } from "@/lib/utils";
import { usePathname } from "next/navigation";
import { useEffect, useRef } from "react";

export function useArtifactsPanelNavCollapse() {
  const pathname = usePathname();
  const { open, setOpen } = useSidebar();
  const isCopilotRoute = matchesRoute(pathname, "/copilot");
  // The docked right panel: an open artifact preview or the artifacts tab.
  // The floating files card doesn't count — it overlays the chat column
  // without narrowing it. Leaving /copilot counts as closing, so the nav
  // comes back when the user navigates away with the panel still open.
  const isPanelOpen = useCopilotUIStore(
    (s) =>
      s.artifactPanel.activeArtifact != null ||
      (s.artifactPanel.isOpen && s.artifactPanel.activeTab === "artifacts"),
  );
  const collapsesNav = isCopilotRoute && isPanelOpen;

  const openBeforePanelRef = useRef<boolean | null>(null);
  const openRef = useRef(open);
  openRef.current = open;
  // setOpen's identity changes with `open`, so the effect must only act on
  // panel-open transitions — otherwise a manual re-open of the nav while the
  // panel is up would snap it shut again.
  const wasCollapsedRef = useRef(false);

  useEffect(() => {
    if (collapsesNav === wasCollapsedRef.current) return;
    wasCollapsedRef.current = collapsesNav;

    if (collapsesNav) {
      openBeforePanelRef.current = openRef.current;
      setOpen(false);
      return;
    }
    // Restore only what this hook collapsed. If the user re-opened the nav
    // by hand while the panel was up, that is a deliberate choice and closing
    // it again on panel-close would undo it in front of them.
    if (openBeforePanelRef.current !== null && openRef.current === false) {
      setOpen(openBeforePanelRef.current);
    }
    openBeforePanelRef.current = null;
  }, [collapsesNav, setOpen]);
}
