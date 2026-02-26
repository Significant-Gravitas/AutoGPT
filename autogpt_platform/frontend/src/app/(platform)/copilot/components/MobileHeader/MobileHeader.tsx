import { Button } from "@/components/atoms/Button/Button";
import { NAVBAR_HEIGHT_PX } from "@/lib/constants";
import { ListIcon } from "@phosphor-icons/react";

interface Props {
  onOpenDrawer: () => void;
}

export function MobileHeader({ onOpenDrawer }: Props) {
  return (
    <div
      className="fixed z-50 flex gap-2"
      style={{ left: "1rem", top: `${NAVBAR_HEIGHT_PX + 20}px` }}
    >
      <Button
        variant="icon"
        size="icon"
        aria-label="Open sessions"
        onClick={onOpenDrawer}
        className="bg-white shadow-md"
      >
        <ListIcon width="1.25rem" height="1.25rem" />
      </Button>
    </div>
  );
}
