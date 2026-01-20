import { Button } from "@/components/atoms/Button/Button";
import { NAVBAR_HEIGHT_PX } from "@/lib/constants";
import { ListIcon } from "@phosphor-icons/react";

interface Props {
  onOpenDrawer: () => void;
}

export function MobileHeader({ onOpenDrawer }: Props) {
  return (
    <Button
      variant="icon"
      size="icon"
      aria-label="Open sessions"
      onClick={onOpenDrawer}
      className="fixed z-50 bg-white shadow-md"
      style={{ left: "1rem", top: `${NAVBAR_HEIGHT_PX + 20}px` }}
    >
      <ListIcon width="1.25rem" height="1.25rem" />
    </Button>
  );
}
