import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { Dialog } from "@/components/molecules/Dialog/Dialog";

interface Props {
  isOpen: boolean;
  title: string;
  onCancel: () => void;
  onConfirm: () => void;
}

export function DeleteTaskDialog({
  isOpen,
  title,
  onCancel,
  onConfirm,
}: Props) {
  return (
    <Dialog
      title="Delete this task?"
      variant="compact"
      className="max-w-md max-lg:inset-x-0 max-lg:mx-auto"
      controlled={{
        isOpen,
        set: (open) => {
          if (!open) onCancel();
        },
      }}
    >
      <Dialog.Content>
        <Text variant="body" tone="text" className="break-words">
          “{title}” and its activity go away for good. Any work in progress
          stops.
        </Text>
        <Dialog.Footer>
          <Button variant="ghost-muted" size="small" onClick={onCancel}>
            Keep it
          </Button>
          <Button variant="destructive" size="small" onClick={onConfirm}>
            Delete task
          </Button>
        </Dialog.Footer>
      </Dialog.Content>
    </Dialog>
  );
}
