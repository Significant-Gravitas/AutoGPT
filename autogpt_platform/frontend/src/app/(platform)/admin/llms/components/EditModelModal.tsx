"use client";

import { useState } from "react";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { Button } from "@/components/atoms/Button/Button";
import type { LlmModel, LlmProvider } from "@/lib/autogpt-server-api/types";
import { updateLlmModelAction } from "../actions";

export function EditModelModal({
  model,
  providers,
}: {
  model: LlmModel;
  providers: LlmProvider[];
}) {
  const [open, setOpen] = useState(false);
  const cost = model.costs[0];
  const provider = providers.find((p) => p.id === model.provider_id);

  return (
    <Dialog
      title="Edit Model"
      controlled={{ isOpen: open, set: setOpen }}
      styling={{ maxWidth: "768px", maxHeight: "90vh", overflowY: "auto" }}
    >
      <Dialog.Trigger>
        <Button variant="outline" size="small" className="min-w-0">
          Edit
        </Button>
      </Dialog.Trigger>
      <Dialog.Content>
        <div className="mb-4 text-sm text-muted-foreground">
          Update model metadata and pricing information.
        </div>
        <form
          action={async (formData) => {
            await updateLlmModelAction(formData);
            setOpen(false);
          }}
          className="space-y-4"
        >
          <input type="hidden" name="model_id" value={model.id} />
          
          <div className="grid gap-4 md:grid-cols-2">
            <label className="text-sm font-medium">
              Display Name
              <input
                required
                name="display_name"
                defaultValue={model.display_name}
                className="mt-1 w-full rounded border border-input bg-background p-2 text-sm"
              />
            </label>
            <label className="text-sm font-medium">
              Provider
              <select
                required
                name="provider_id"
                className="mt-1 w-full rounded border border-input bg-background p-2 text-sm"
                defaultValue={model.provider_id}
              >
                {providers.map((p) => (
                  <option key={p.id} value={p.id}>
                    {p.display_name} ({p.name})
                  </option>
                ))}
              </select>
            </label>
          </div>

          <label className="text-sm font-medium">
            Description
            <textarea
              name="description"
              rows={2}
              defaultValue={model.description ?? ""}
              className="mt-1 w-full rounded border border-input bg-background p-2 text-sm"
              placeholder="Optional description..."
            />
          </label>

          <div className="grid gap-4 md:grid-cols-2">
            <label className="text-sm font-medium">
              Context Window
              <input
                required
                type="number"
                name="context_window"
                defaultValue={model.context_window}
                className="mt-1 w-full rounded border border-input bg-background p-2 text-sm"
                min={1}
              />
            </label>
            <label className="text-sm font-medium">
              Max Output Tokens
              <input
                type="number"
                name="max_output_tokens"
                defaultValue={model.max_output_tokens ?? undefined}
                className="mt-1 w-full rounded border border-input bg-background p-2 text-sm"
                min={1}
              />
            </label>
          </div>

          <div className="grid gap-4 md:grid-cols-4">
            <label className="text-sm font-medium">
              Credit Cost
              <input
                required
                type="number"
                name="credit_cost"
                defaultValue={cost?.credit_cost ?? 0}
                className="mt-1 w-full rounded border border-input bg-background p-2 text-sm"
                min={0}
              />
            </label>
            <label className="text-sm font-medium">
              Credential Provider
              <input
                required
                name="credential_provider"
                defaultValue={cost?.credential_provider ?? ""}
                className="mt-1 w-full rounded border border-input bg-background p-2 text-sm"
              />
            </label>
            <label className="text-sm font-medium">
              Credential ID
              <input
                name="credential_id"
                defaultValue={cost?.credential_id ?? ""}
                className="mt-1 w-full rounded border border-input bg-background p-2 text-sm"
                placeholder="cred-id"
              />
            </label>
            <label className="text-sm font-medium">
              Credential Type
              <input
                name="credential_type"
                defaultValue={cost?.credential_type ?? "api_key"}
                className="mt-1 w-full rounded border border-input bg-background p-2 text-sm"
              />
            </label>
          </div>

          <label className="flex items-center gap-2 text-sm font-medium">
            <input type="hidden" name="is_enabled" value="off" />
            <input
              type="checkbox"
              name="is_enabled"
              defaultChecked={model.is_enabled}
            />
            Enabled
          </label>

          <Dialog.Footer>
            <Button
              variant="ghost"
              size="small"
              onClick={() => setOpen(false)}
            >
              Cancel
            </Button>
            <Button variant="primary" size="small" type="submit">
              Update Model
            </Button>
          </Dialog.Footer>
        </form>
      </Dialog.Content>
    </Dialog>
  );
}

