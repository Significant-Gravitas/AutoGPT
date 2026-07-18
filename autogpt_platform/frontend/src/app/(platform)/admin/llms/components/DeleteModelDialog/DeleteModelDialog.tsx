"use client";

import type { LlmModelAdminResponse } from "@/app/api/__generated__/models/llmModelAdminResponse";
import { Button } from "@/components/atoms/Button/Button";
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

const NO_REPLACEMENT = "__none__";

export function DeleteModelDialog({ open, model, models, onClose }: Props) {
  const { deleteModel } = useLlmRegistryMutations();
  const [replacement, setReplacement] = useState(
    model?.fallback_model_slug ?? NO_REPLACEMENT,
  );

  if (!model) return null;

  async function handleConfirm() {
    if (!model) return;
    await deleteModel.mutateAsync({
      slug: model.slug,
      params:
        replacement === NO_REPLACEMENT
          ? undefined
          : { replacement_model_slug: replacement },
    });
    onClose();
  }

  return (
    <Dialog
      title={`Delete ${model.slug}`}
      styling={{ maxWidth: "34rem" }}
      controlled={{
        isOpen: open,
        set: (next) => (next ? undefined : onClose()),
      }}
    >
      <Dialog.Content>
        <div className="flex flex-col gap-3">
          <p className="text-sm text-muted-foreground">
            Deletion is permanent. If agent nodes still reference this model,
            the API requires a replacement model to migrate them to — prefer
            disabling unless the row was added in error.
          </p>
          <Select
            id="delete-replacement"
            label="Replacement model"
            value={replacement}
            onValueChange={setReplacement}
            options={[
              { value: NO_REPLACEMENT, label: "— none (fails if in use) —" },
              ...models
                .filter((m) => m.slug !== model.slug && m.is_enabled)
                .map((m) => ({ value: m.slug, label: m.slug })),
            ]}
          />
        </div>
        <Dialog.Footer>
          <Button
            variant="secondary"
            onClick={onClose}
            disabled={deleteModel.isPending}
          >
            Cancel
          </Button>
          <Button
            variant="destructive"
            onClick={handleConfirm}
            loading={deleteModel.isPending}
          >
            Delete model
          </Button>
        </Dialog.Footer>
      </Dialog.Content>
    </Dialog>
  );
}
