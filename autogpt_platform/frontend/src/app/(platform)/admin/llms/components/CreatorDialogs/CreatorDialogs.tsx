"use client";

import type { LlmCreatorAdminResponse } from "@/app/api/__generated__/models/llmCreatorAdminResponse";
import { Button } from "@/components/atoms/Button/Button";
import { Input } from "@/components/atoms/Input/Input";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { useState } from "react";
import { useLlmRegistryMutations } from "../useLlmRegistryMutations";

interface FormProps {
  open: boolean;
  editing: LlmCreatorAdminResponse | null;
  onClose: () => void;
}

export function CreatorFormDialog({ open, editing, onClose }: FormProps) {
  const { createCreator, updateCreator } = useLlmRegistryMutations();
  const [name, setName] = useState(editing?.name ?? "");
  const [displayName, setDisplayName] = useState(editing?.display_name ?? "");
  const [websiteUrl, setWebsiteUrl] = useState(editing?.website_url ?? "");

  const isPending = createCreator.isPending || updateCreator.isPending;

  async function handleSubmit() {
    if (editing) {
      await updateCreator.mutateAsync({
        name: editing.name,
        data: {
          display_name: displayName,
          website_url: websiteUrl || null,
        },
      });
    } else {
      await createCreator.mutateAsync({
        data: {
          name,
          display_name: displayName,
          website_url: websiteUrl || null,
        },
      });
    }
    onClose();
  }

  return (
    <Dialog
      title={editing ? `Edit ${editing.name}` : "Add creator"}
      styling={{ maxWidth: "30rem" }}
      controlled={{
        isOpen: open,
        set: (next) => (next ? undefined : onClose()),
      }}
    >
      <Dialog.Content>
        <div className="flex flex-col gap-3">
          {!editing && (
            <Input
              id="creator-name"
              label="Name (slug)"
              placeholder="e.g. moonshotai"
              value={name}
              onChange={(e) => setName(e.target.value)}
            />
          )}
          <Input
            id="creator-display-name"
            label="Display name"
            value={displayName}
            onChange={(e) => setDisplayName(e.target.value)}
          />
          <Input
            id="creator-website"
            label="Website URL"
            value={websiteUrl}
            onChange={(e) => setWebsiteUrl(e.target.value)}
          />
        </div>
        <Dialog.Footer>
          <Button variant="secondary" onClick={onClose} disabled={isPending}>
            Cancel
          </Button>
          <Button onClick={handleSubmit} loading={isPending}>
            {editing ? "Save changes" : "Create creator"}
          </Button>
        </Dialog.Footer>
      </Dialog.Content>
    </Dialog>
  );
}

interface DeleteProps {
  open: boolean;
  creator: LlmCreatorAdminResponse | null;
  onClose: () => void;
}

export function DeleteCreatorDialog({ open, creator, onClose }: DeleteProps) {
  const { deleteCreator } = useLlmRegistryMutations();

  if (!creator) return null;

  async function handleConfirm() {
    if (!creator) return;
    await deleteCreator.mutateAsync({ name: creator.name });
    onClose();
  }

  return (
    <Dialog
      title={`Delete ${creator.name}?`}
      styling={{ maxWidth: "28rem" }}
      controlled={{
        isOpen: open,
        set: (next) => (next ? undefined : onClose()),
      }}
    >
      <Dialog.Content>
        <p className="text-sm text-muted-foreground">
          Models referencing this creator keep working (creator becomes unset).
          This cannot be undone.
        </p>
        <Dialog.Footer>
          <Button
            variant="secondary"
            onClick={onClose}
            disabled={deleteCreator.isPending}
          >
            Cancel
          </Button>
          <Button
            variant="destructive"
            onClick={handleConfirm}
            loading={deleteCreator.isPending}
          >
            Delete creator
          </Button>
        </Dialog.Footer>
      </Dialog.Content>
    </Dialog>
  );
}
