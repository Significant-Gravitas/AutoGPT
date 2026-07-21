import { Button } from "@/components/atoms/Button/Button";
import { NAVBAR_HEIGHT_PX } from "@/lib/constants";
import { Flag, useGetFlag } from "@/services/feature-flags/use-get-flag";
import { FolderIcon, ListIcon } from "@phosphor-icons/react";
import { useCopilotUIStore } from "../../store";

export function MobileHeader() {
  const setDrawerOpen = useCopilotUIStore((s) => s.setDrawerOpen);
  const toggleContextPanel = useCopilotUIStore((s) => s.toggleContextPanel);
  const isContextPanelEnabled = useGetFlag(Flag.ARTIFACTS);
  return (
    <div
      className="fixed z-50 flex gap-2"
      style={{ left: "1rem", top: `${NAVBAR_HEIGHT_PX + 20}px` }}
    >
      <Button
        variant="icon"
        size="icon"
        aria-label="Open sessions"
        onClick={() => setDrawerOpen(true)}
        className="bg-white shadow-md"
      >
        <ListIcon width="1.25rem" height="1.25rem" />
      </Button>
      {isContextPanelEnabled && (
        <Button
          variant="icon"
          size="icon"
          aria-label="Open workspace files"
          onClick={toggleContextPanel}
          className="bg-white shadow-md"
        >
          <FolderIcon width="1.25rem" height="1.25rem" />
        </Button>
      )}
    </div>
  );
}
