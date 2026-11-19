import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { Dialog } from "@/components/molecules/Dialog/Dialog";

interface Props {
  isOpen: boolean;
  title: string;
  isPending: boolean;
  onCancel: () => void;
  onConfirm: () => void;
  /** A repeating task: only the run going now stops. */
  isRepeating?: boolean;
}

/** Asks before stopping a task's work: what it was doing is cut off. */
export function StopTaskDialog({
  isOpen,
  title,
  isPending,
  onCancel,
  onConfirm,
  isRepeating = false,
}: Props) {
  return (
    <Dialog
      title={isRepeating ? "Stop this run?" : "Stop this task?"}
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
          The work on “<bdi>{title}</bdi>” stops where it is.{" "}
          {isRepeating
            ? "It still runs again at its next time."
            : "You can try it again later."}
        </Text>
        <Dialog.Footer>
          <Button variant="ghost-muted" size="small" onClick={onCancel}>
            Keep going
          </Button>
          <Button
            variant="destructive"
            size="small"
            loading={isPending}
            onClick={onConfirm}
          >
            {isRepeating ? "Stop run" : "Stop task"}
          </Button>
        </Dialog.Footer>
      </Dialog.Content>
    </Dialog>
  );
}
