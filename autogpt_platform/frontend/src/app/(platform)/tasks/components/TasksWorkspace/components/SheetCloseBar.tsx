import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { Cancel01Icon } from "@hugeicons/core-free-icons";

interface Props {
  onClose: () => void;
}

/** Below lg the task opens as a full-screen sheet, which has no close
 *  button of its own; the side panel above lg brings its own. */
export function SheetCloseBar({ onClose }: Props) {
  return (
    <div className="flex h-12 shrink-0 items-center justify-between border-b border-border-hairline px-4 lg:hidden">
      <Text variant="body-medium" as="span" tone="ink">
        Task
      </Text>
      <Button
        type="button"
        variant="ghost-muted"
        size="icon-sm"
        leadingIcon={Cancel01Icon}
        aria-label="Close task"
        withTooltip={false}
        onClick={onClose}
      />
    </div>
  );
}
