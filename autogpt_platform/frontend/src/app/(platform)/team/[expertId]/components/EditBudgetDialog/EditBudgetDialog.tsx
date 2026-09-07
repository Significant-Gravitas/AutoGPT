"use client";

import { Expert } from "@/app/api/__generated__/models/expert";
import { Button } from "@/components/atoms/Button/Button";
import { Input } from "@/components/atoms/Input/Input";
import { Text } from "@/components/atoms/Text/Text";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { useEditBudgetDialog } from "./useEditBudgetDialog";

interface Props {
  expert: Expert;
  open: boolean;
  onClose: () => void;
}

export function EditBudgetDialog({ expert, open, onClose }: Props) {
  const { value, setValue, isInvalid, isPending, save } = useEditBudgetDialog({
    expert,
    open,
    onClose,
  });

  return (
    <Dialog
      variant="compact"
      title="Weekly budget"
      styling={{ maxWidth: "24rem" }}
      controlled={{
        isOpen: open,
        set: (next) => {
          if (!next) onClose();
        },
      }}
    >
      <Dialog.Content>
        <form onSubmit={save} className="flex flex-col gap-4 px-1">
          <div className="flex flex-col gap-1.5">
            <Input
              id="expert-weekly-budget"
              size="small"
              label="Weekly budget"
              labelVariant="small-medium"
              placeholder="e.g. 25"
              value={value}
              error={
                isInvalid ? "Enter a dollar amount like 25 or 12.50" : undefined
              }
              onChange={(event) => setValue(event.target.value)}
              wrapperClassName="!mb-0"
            />
            <Text variant="small" tone="muted">
              What {expert.name} may spend each week, in dollars. Leave empty
              for the default, or enter 0 to remove the cap.
            </Text>
          </div>
          <div className="flex justify-end gap-2">
            <Button
              size="xs"
              type="button"
              variant="secondary"
              onClick={onClose}
            >
              Cancel
            </Button>
            <Button
              size="xs"
              type="submit"
              variant="primary"
              disabled={isInvalid}
              loading={isPending}
            >
              Save budget
            </Button>
          </div>
        </form>
      </Dialog.Content>
    </Dialog>
  );
}
