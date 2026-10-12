import type { BoardColumn } from "@/app/api/__generated__/models/boardColumn";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { Dialog } from "@/components/molecules/Dialog/Dialog";

interface Props {
  column: BoardColumn;
  onConfirm: () => void;
  onCancel: () => void;
}

export function DeleteColumnDialog({ column, onConfirm, onCancel }: Props) {
  return (
    <Dialog
      title={`Delete “${column.name}”?`}
      styling={{ maxWidth: "28rem" }}
      controlled={{
        isOpen: true,
        set: (open) => {
          if (!open) onCancel();
        },
      }}
    >
      <Dialog.Content>
        <div className="flex flex-col gap-4 px-1">
          <Text variant="body" className="text-zinc-700">
            Its cards move to Backlog.
          </Text>
          <div className="flex justify-end gap-2 pt-2">
            <Button variant="secondary" onClick={onCancel}>
              Cancel
            </Button>
            <Button variant="destructive" onClick={onConfirm}>
              Delete
            </Button>
          </div>
        </div>
      </Dialog.Content>
    </Dialog>
  );
}
