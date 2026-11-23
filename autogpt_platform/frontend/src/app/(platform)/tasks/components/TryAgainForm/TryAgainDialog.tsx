import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { TryAgainForm } from "./TryAgainForm";

interface Props {
  isOpen: boolean;
  taskID: string;
  isPending: boolean;
  onCancel: () => void;
  onSubmit: (instructions: string) => void;
}

/** The task menu's Try again: the same optional note, in a small dialog. */
export function TryAgainDialog({
  isOpen,
  taskID,
  isPending,
  onCancel,
  onSubmit,
}: Props) {
  return (
    <Dialog
      title="Try this task again"
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
        <TryAgainForm
          taskID={taskID}
          isPending={isPending}
          onSubmit={onSubmit}
          onCancel={onCancel}
        />
      </Dialog.Content>
    </Dialog>
  );
}
