"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Input } from "@/components/atoms/Input/Input";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { isKey } from "@/lib/keyboard";
import type { SavedFileRef } from "../../helpers";
import { useSaveViewDialog } from "./useSaveViewDialog";

interface Props {
  ownerId: string | undefined;
  sessionId: string;
  viewId: string;
  title: string;
  beforeSave: () => Promise<void>;
  onClose: () => void;
  onSaved: (file: SavedFileRef) => void;
}

export function SaveViewDialog({ onClose, ...args }: Props) {
  const { name, setName, error, canSubmit, isPending, handleSubmit } =
    useSaveViewDialog(args);

  return (
    <Dialog
      controlled={{
        isOpen: true,
        set: (open) => {
          if (!open) onClose();
        },
      }}
      styling={{ maxWidth: "28rem" }}
      title="Save to files"
    >
      <Dialog.Content>
        <div className="flex flex-col gap-3">
          <Input
            id="save-view-file-name"
            label="File name"
            placeholder="Enter file name"
            value={name}
            error={error ?? undefined}
            onChange={(e) => setName(e.target.value)}
            onKeyDown={(e) => {
              if (isKey(e, "Enter")) void handleSubmit();
            }}
            className="w-full"
            wrapperClassName="!mb-0"
          />
          <Button
            variant="primary"
            className="mt-2"
            disabled={!canSubmit}
            loading={isPending}
            onClick={() => void handleSubmit()}
          >
            Save
          </Button>
        </div>
      </Dialog.Content>
    </Dialog>
  );
}
