"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { Button } from "@/components/atoms/Button/Button";
import type { LlmModel } from "@/app/api/__generated__/models/llmModel";
import type { LlmModelCreator } from "@/app/api/__generated__/models/llmModelCreator";
import type { LlmProvider } from "@/app/api/__generated__/models/llmProvider";
import { updateLlmModelAction } from "../actions";

export function EditModelModal({
  model,
  providers,
  creators,
}: {
  model: LlmModel;
  providers: LlmProvider[];
  creators: LlmModelCreator[];
}) {
  const router = useRouter();
  const [open, setOpen] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const cost = model.costs?.[0];
  const provider = providers.find((p) => p.id === model.provider_id);

  async function handleSubmit(formData: FormData) {
    setIsSubmitting(true);
    setError(null);
    try {
      await updateLlmModelAction(formData);
      setOpen(false);
      router.refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to update model");
    } finally {
      setIsSubmitting(false);
    }
  }

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
        {error && (
          <div className="mb-4 rounded-lg border border-destructive/30 bg-destructive/10 p-3 text-sm text-destructive">
            {error}
          </div>
        )}
        <form action={handleSubmit} className="space-y-4">
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
              <span className="text-xs text-muted-foreground">
                Who hosts/serves the model
              </span>
            </label>
          </div>

          <div className="grid gap-4 md:grid-cols-2">
            <label className="text-sm font-medium">
              Creator
              <select
                name="creator_id"
                className="mt-1 w-full rounded border border-input bg-background p-2 text-sm"
                defaultValue={model.creator_id ?? ""}
              >
                <option value="">No creator selected</option>
                {creators.map((c) => (
                  <option key={c.id} value={c.id}>
                    {c.display_name} ({c.name})
                  </option>
                ))}
              </select>
              <span className="text-xs text-muted-foreground">
                Who made/trained the model (e.g., OpenAI, Meta)
              </span>
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

          <div className="grid gap-4 md:grid-cols-2">
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
              <span className="text-xs text-muted-foreground">
                Credits charged per run
              </span>
            </label>
            <label className="text-sm font-medium">
              Credential Provider
              <select
                required
                name="credential_provider"
                defaultValue={cost?.credential_provider ?? provider?.name ?? ""}
                className="mt-1 w-full rounded border border-input bg-background p-2 text-sm"
              >
                <option value="" disabled>
                  Select provider
                </option>
                {providers.map((p) => (
                  <option key={p.id} value={p.name}>
                    {p.display_name} ({p.name})
                  </option>
                ))}
              </select>
              <span className="text-xs text-muted-foreground">
                Must match a key in PROVIDER_CREDENTIALS
              </span>
            </label>
          </div>
          {/* Hidden defaults for credential_type and unit */}
          <input
            type="hidden"
            name="credential_type"
            value={
              cost?.credential_type ??
              provider?.default_credential_type ??
              "api_key"
            }
          />
          <input type="hidden" name="unit" value={cost?.unit ?? "RUN"} />

          <Dialog.Footer>
            <Button
              type="button"
              variant="ghost"
              size="small"
              onClick={() => setOpen(false)}
              disabled={isSubmitting}
            >
              Cancel
            </Button>
            <Button
              variant="primary"
              size="small"
              type="submit"
              disabled={isSubmitting}
            >
              {isSubmitting ? "Updating..." : "Update Model"}
            </Button>
          </Dialog.Footer>
        </form>
      </Dialog.Content>
    </Dialog>
  );
}
