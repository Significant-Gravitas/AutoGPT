import { Button } from "@/components/atoms/Button/Button";
import { NAVBAR_HEIGHT_PX } from "@/lib/constants";
import { useCopilotUIStore } from "../../store";
import { Folder01Icon, Menu01Icon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";
import { cn } from "@/lib/utils";

interface Props {
  embedded?: boolean;
}

export function MobileHeader({ embedded = false }: Props) {
  const setDrawerOpen = useCopilotUIStore((s) => s.setDrawerOpen);
  const toggleContextPanel = useCopilotUIStore((s) => s.toggleContextPanel);
  return (
    <div
      className={cn(
        "z-50 flex gap-2",
        embedded ? "shrink-0 px-4 pb-1 pt-3" : "fixed",
      )}
      style={
        embedded
          ? undefined
          : { left: "1rem", top: `${NAVBAR_HEIGHT_PX + 20}px` }
      }
    >
      <Button
        variant="icon"
        size="icon"
        aria-label="Open sessions"
        onClick={() => setDrawerOpen(true)}
        className="bg-white shadow-md"
      >
        <Icon icon={Menu01Icon} width="1.25rem" height="1.25rem" />
      </Button>
      <Button
        variant="icon"
        size="icon"
        aria-label="Open workspace files"
        onClick={toggleContextPanel}
        className="bg-white shadow-md"
      >
        <Icon icon={Folder01Icon} width="1.25rem" height="1.25rem" />
      </Button>
    </div>
  );
}
