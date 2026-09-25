"use client";

import { useEffect, useState } from "react";
import { Button } from "@/components/atoms/Button/Button";
import { Input } from "@/components/atoms/Input/Input";
import { Text } from "@/components/atoms/Text/Text";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { isKey } from "@/lib/keyboard";

interface Props {
  isOpen: boolean;
  setIsOpen: (open: boolean) => void;
  mode: "create" | "edit";
  initialName?: string;
  /** Folder the new one lands in, shown under the input. Absent at the root. */
  location?: string;
  isSubmitting: boolean;
  onSubmit: (values: { name: string }) => void;
}

export function FolderFormDialog({
  isOpen,
  setIsOpen,
  mode,
  initialName = "",
  location,
  isSubmitting,
  onSubmit,
}: Props) {
  const [name, setName] = useState(initialName);

  useEffect(() => {
    if (isOpen) {
      setName(initialName);
    }
  }, [isOpen, initialName]);

  const trimmed = name.trim();
  const canSubmit =
    trimmed.length > 0 && trimmed.length <= 100 && !isSubmitting;

  function handleSubmit() {
    if (!canSubmit) return;
    onSubmit({ name: trimmed });
  }

  return (
    <Dialog
      controlled={{ isOpen, set: setIsOpen }}
      styling={{ maxWidth: "28rem" }}
      title={mode === "create" ? "Create folder" : "Rename folder"}
    >
      <Dialog.Content>
        <div className="flex flex-col gap-3">
          <Input
            id="folder-name"
            label="Folder name"
            placeholder="Enter folder name"
            value={name}
            onChange={(e) => setName(e.target.value)}
            onKeyDown={(e) => {
              if (isKey(e, "Enter")) handleSubmit();
            }}
            className="w-full"
            wrapperClassName="!mb-0"
          />
          {location ? (
            <Text variant="small" className="text-zinc-500">
              Inside &ldquo;{location}&rdquo;
            </Text>
          ) : null}
          <Button
            variant="primary"
            className="mt-2"
            disabled={!canSubmit}
            loading={isSubmitting}
            onClick={handleSubmit}
            data-testid="folder-form-submit"
          >
            {mode === "create" ? "Create" : "Save"}
          </Button>
        </div>
      </Dialog.Content>
    </Dialog>
  );
}
