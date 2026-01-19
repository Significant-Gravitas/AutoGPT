import { Button } from "@/components/atoms/Button/Button";
import { List } from "@phosphor-icons/react";

interface Props {
  onOpenDrawer: () => void;
}

export function MobileHeader({ onOpenDrawer }: Props) {
  return (
    <header className="flex items-center justify-between px-4 py-3">
      <Button
        variant="icon"
        size="icon"
        aria-label="Open sessions"
        onClick={onOpenDrawer}
      >
        <List width="1.25rem" height="1.25rem" />
      </Button>
    </header>
  );
}
