"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Input } from "@/components/atoms/Input/Input";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { FormEvent, useEffect, useState } from "react";

interface Props {
  open: boolean;
  onClose: () => void;
  onCreate: (name: string) => void;
  isCreating: boolean;
}

export function NewPodDialog({ open, onClose, onCreate, isCreating }: Props) {
  const [name, setName] = useState("");
  const trimmed = name.trim();

  useEffect(() => {
    if (!open) setName("");
  }, [open]);

  function handleSubmit(event: FormEvent) {
    event.preventDefault();
    if (!trimmed) return;
    onCreate(trimmed);
  }

  return (
    <Dialog
      title="New pod"
      styling={{ maxWidth: "28rem" }}
      controlled={{
        isOpen: open,
        set: (next) => {
          if (!next) onClose();
        },
      }}
    >
      <Dialog.Content>
        <form onSubmit={handleSubmit} className="flex flex-col gap-4 px-1">
          <Input
            id="pod-name"
            label="Pod name"
            placeholder="e.g. Growth"
            value={name}
            onChange={(event) => setName(event.target.value)}
            wrapperClassName="!mb-0"
          />
          <div className="flex justify-end gap-2">
            <Button type="button" variant="secondary" onClick={onClose}>
              Cancel
            </Button>
            <Button
              type="submit"
              variant="primary"
              disabled={!trimmed || isCreating}
              loading={isCreating}
            >
              Create pod
            </Button>
          </div>
        </form>
      </Dialog.Content>
    </Dialog>
  );
}
