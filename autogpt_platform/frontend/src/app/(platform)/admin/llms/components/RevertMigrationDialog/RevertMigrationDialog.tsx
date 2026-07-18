"use client";

import type { LlmMigrationAdminResponse } from "@/app/api/__generated__/models/llmMigrationAdminResponse";
import { Button } from "@/components/atoms/Button/Button";
import { Switch } from "@/components/atoms/Switch/Switch";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { useState } from "react";
import { useLlmRegistryMutations } from "../useLlmRegistryMutations";

interface Props {
  open: boolean;
  migration: LlmMigrationAdminResponse | null;
  onClose: () => void;
}

export function RevertMigrationDialog({ open, migration, onClose }: Props) {
  const { revertMigration } = useLlmRegistryMutations();
  const [reEnableSource, setReEnableSource] = useState(true);

  if (!migration) return null;

  async function handleConfirm() {
    if (!migration) return;
    await revertMigration.mutateAsync({
      migrationId: migration.id,
      params: { re_enable_source_model: reEnableSource },
    });
    onClose();
  }

  return (
    <Dialog
      title="Revert migration?"
      styling={{ maxWidth: "30rem" }}
      controlled={{
        isOpen: open,
        set: (next) => (next ? undefined : onClose()),
      }}
    >
      <Dialog.Content>
        <div className="flex flex-col gap-3">
          <p className="text-sm text-muted-foreground">
            Moves {migration.node_count} node(s) back from{" "}
            <code className="text-xs">{migration.target_model_slug}</code> to{" "}
            <code className="text-xs">{migration.source_model_slug}</code>.
          </p>
          <label className="flex items-center gap-2 text-sm">
            <Switch
              checked={reEnableSource}
              onCheckedChange={setReEnableSource}
              aria-label="Re-enable source model"
            />
            Re-enable the source model
          </label>
        </div>
        <Dialog.Footer>
          <Button
            variant="secondary"
            onClick={onClose}
            disabled={revertMigration.isPending}
          >
            Cancel
          </Button>
          <Button onClick={handleConfirm} loading={revertMigration.isPending}>
            Revert
          </Button>
        </Dialog.Footer>
      </Dialog.Content>
    </Dialog>
  );
}
