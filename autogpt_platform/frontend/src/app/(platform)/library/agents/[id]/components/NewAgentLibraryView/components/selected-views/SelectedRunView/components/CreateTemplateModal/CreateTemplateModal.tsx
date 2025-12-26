"use client";

import { GraphExecution } from "@/app/api/__generated__/models/graphExecution";
import { Button } from "@/components/atoms/Button/Button";
import { Input } from "@/components/atoms/Input/Input";
import { Text } from "@/components/atoms/Text/Text";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { useState } from "react";

interface Props {
  isOpen: boolean;
  onClose: () => void;
  onCreate: (name: string, description: string) => Promise<void>;
  run?: GraphExecution;
}

export function CreateTemplateModal({ isOpen, onClose, onCreate }: Props) {
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [isCreating, setIsCreating] = useState(false);

  async function handleSubmit() {
    if (!name.trim()) return;

    setIsCreating(true);
    try {
      await onCreate(name.trim(), description.trim());
      setName("");
      setDescription("");
      onClose();
    } finally {
      setIsCreating(false);
    }
  }

  function handleCancel() {
    setName("");
    setDescription("");
    onClose();
  }

  return (
    <Dialog
      controlled={{ isOpen, set: () => onClose() }}
      styling={{ maxWidth: "500px" }}
    >
      <Dialog.Content>
        <div className="flex flex-col gap-4">
          <div className="flex flex-col gap-1">
            <Text variant="lead" as="h2" className="!font-medium !text-black">
              Create Template
            </Text>
            <Text variant="body" className="text-zinc-600">
              Save this task as a template to reuse later with the same inputs
              and credentials.
            </Text>
          </div>

          <div className="flex w-[96%] flex-col gap-4 pl-1">
            <Input
              id="template-name"
              label="Name"
              placeholder="Enter template name"
              value={name}
              onChange={(e) => setName(e.target.value)}
              autoFocus
            />
            <Input
              type="textarea"
              id="template-description"
              label="Description"
              placeholder="Optional description"
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              rows={3}
            />
          </div>
        </div>

        <Dialog.Footer className="mt-6">
          <div className="flex justify-end gap-3">
            <Button variant="secondary" onClick={handleCancel}>
              Cancel
            </Button>
            <Button
              variant="primary"
              onClick={handleSubmit}
              disabled={!name.trim() || isCreating}
              loading={isCreating}
            >
              Create Template
            </Button>
          </div>
        </Dialog.Footer>
      </Dialog.Content>
    </Dialog>
  );
}
