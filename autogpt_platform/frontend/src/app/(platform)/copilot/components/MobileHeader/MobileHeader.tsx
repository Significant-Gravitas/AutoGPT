import { Button } from "@/components/atoms/Button/Button";
import { NAVBAR_HEIGHT_PX } from "@/lib/constants";
import { ListIcon, TrashIcon } from "@phosphor-icons/react";

interface Props {
  onOpenDrawer: () => void;
  showDelete?: boolean;
  isDeleting?: boolean;
  onDelete?: () => void;
}

export function MobileHeader({
  onOpenDrawer,
  showDelete,
  isDeleting,
  onDelete,
}: Props) {
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
      {showDelete && onDelete && (
        <Button
          variant="icon"
          size="icon"
          aria-label="Delete current chat"
          onClick={onDelete}
          disabled={isDeleting}
          className="bg-white text-red-500 shadow-md hover:bg-red-50 hover:text-red-600 disabled:opacity-50"
        >
          <TrashIcon width="1.25rem" height="1.25rem" />
        </Button>
      )}
    </div>
  );
}
