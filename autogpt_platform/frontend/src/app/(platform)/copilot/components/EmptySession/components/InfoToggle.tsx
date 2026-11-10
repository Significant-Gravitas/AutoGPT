import { Button } from "@/components/atoms/Button/Button";
import { cn } from "@/lib/utils";
import { SidebarRightIcon } from "@hugeicons/core-free-icons";

interface Props {
  isPanelOpen: boolean;
  onOpen: () => void;
}

// Fades out while the panel it opens is on screen, and back in after it
// closes, so the two never sit in the same corner at once.
export function InfoToggle({ isPanelOpen, onOpen }: Props) {
  return (
    <div
      inert={isPanelOpen}
      className={cn(
        "absolute top-4 right-4 z-40 transition-opacity motion-reduce:transition-none",
        isPanelOpen
          ? "opacity-0 duration-200"
          : "opacity-100 delay-150 duration-300",
      )}
    >
      <Button
        type="button"
        variant="ghost-muted"
        size="icon-sm"
        leadingIcon={SidebarRightIcon}
        aria-label="Show expert info"
        className="size-7"
        onClick={onOpen}
      />
    </div>
  );
}
