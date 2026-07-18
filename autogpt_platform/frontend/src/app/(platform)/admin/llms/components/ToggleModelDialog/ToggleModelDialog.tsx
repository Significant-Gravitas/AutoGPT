"use client";

import type { LlmModelAdminResponse } from "@/app/api/__generated__/models/llmModelAdminResponse";
import { Button } from "@/components/atoms/Button/Button";
import { Input } from "@/components/atoms/Input/Input";
import { Select } from "@/components/atoms/Select/Select";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { useState } from "react";
import { useLlmRegistryMutations } from "../useLlmRegistryMutations";

interface Props {
  open: boolean;
  model: LlmModelAdminResponse | null;
  models: LlmModelAdminResponse[];
  onClose: () => void;
}

const NO_MIGRATION = "__none__";

export function ToggleModelDialog({ open, model, models, onClose }: Props) {
  const { toggleModel } = useLlmRegistryMutations();
  const [migrateTo, setMigrateTo] = useState(
    model?.fallback_model_slug ?? NO_MIGRATION,
  );
  const [reason, setReason] = useState("");

  if (!model) return null;

  const replacementOptions = models
    .filter((m) => m.slug !== model.slug && m.is_enabled)
    .map((m) => ({ value: m.slug, label: m.slug }));

  async function handleConfirm() {
    if (!model) return;
    await toggleModel.mutateAsync({
      slug: model.slug,
      data: {
        is_enabled: false,
        migrate_to_slug: migrateTo === NO_MIGRATION ? null : migrateTo,
        migration_reason: reason || null,
      },
    });
    onClose();
  }

  return (
    <Dialog
      title={`Disable ${model.slug}`}
      styling={{ maxWidth: "34rem" }}
      controlled={{
        isOpen: open,
        set: (next) => (next ? undefined : onClose()),
      }}
    >
      <Dialog.Content>
        <div className="flex flex-col gap-3">
          <p className="text-sm text-muted-foreground">
            Disabling is the kill switch: the model stops serving everywhere,
            even when LaunchDarkly routes to it. Optionally migrate agent nodes
            that use it to a replacement model.
          </p>
          <Select
            id="toggle-migrate-to"
            label="Migrate existing nodes to"
            value={migrateTo}
            onValueChange={setMigrateTo}
            options={[
              { value: NO_MIGRATION, label: "— no migration —" },
              ...replacementOptions,
            ]}
          />
          <Input
            id="toggle-reason"
            label="Reason"
            placeholder="e.g. provider outage"
            value={reason}
            onChange={(e) => setReason(e.target.value)}
          />
        </div>
        <Dialog.Footer>
          <Button
            variant="secondary"
            onClick={onClose}
            disabled={toggleModel.isPending}
          >
            Cancel
          </Button>
          <Button onClick={handleConfirm} loading={toggleModel.isPending}>
            Disable model
          </Button>
        </Dialog.Footer>
      </Dialog.Content>
    </Dialog>
  );
}
