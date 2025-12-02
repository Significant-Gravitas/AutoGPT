"use client";

import { useState } from "react";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { Button } from "@/components/atoms/Button/Button";
import type { LlmModel } from "@/lib/autogpt-server-api/types";
import { deleteLlmModelAction } from "../actions";

export function DeleteModelModal({
  model,
  availableModels,
}: {
  model: LlmModel;
  availableModels: LlmModel[];
}) {
  const [open, setOpen] = useState(false);
  const [selectedReplacement, setSelectedReplacement] = useState<string>("");
  const [isDeleting, setIsDeleting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [usageCount, setUsageCount] = useState<number | null>(null);

  // Filter out the current model and disabled models from replacement options
  const replacementOptions = availableModels.filter(
    (m) => m.id !== model.id && m.is_enabled
  );

  async function fetchUsage() {
    try {
      const BackendApi = (await import("@/lib/autogpt-server-api")).default;
      const api = new BackendApi();
      const usage = await api.getAdminLlmModelUsage(model.id);
      setUsageCount(usage.node_count);
    } catch {
      setUsageCount(null);
    }
  }

  async function handleDelete(formData: FormData) {
    setIsDeleting(true);
    setError(null);
    try {
      await deleteLlmModelAction(formData);
      setOpen(false);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to delete model");
    } finally {
      setIsDeleting(false);
    }
  }

  return (
    <Dialog
      title="Delete Model"
      controlled={{
        isOpen: open,
        set: async (isOpen) => {
          setOpen(isOpen);
          if (isOpen) {
            setUsageCount(null);
            await fetchUsage();
          }
        },
      }}
      styling={{ maxWidth: "600px" }}
    >
      <Dialog.Trigger>
        <button
          type="button"
          className="inline-flex items-center rounded border border-red-300 px-3 py-1 text-xs font-semibold text-red-600 hover:bg-red-50"
        >
          Delete
        </button>
      </Dialog.Trigger>
      <Dialog.Content>
        <div className="mb-4 text-sm text-muted-foreground">
          This action cannot be undone. All workflows using this model will be
          migrated to the replacement model you select.
        </div>

        <div className="space-y-4">
          <div className="rounded-lg border border-yellow-200 bg-yellow-50 p-4">
            <div className="flex items-start gap-3">
              <div className="flex-shrink-0 text-yellow-600">⚠️</div>
              <div className="text-sm text-yellow-800">
                <p className="font-semibold">You are about to delete:</p>
                <p className="mt-1">
                  <span className="font-medium">{model.display_name}</span>{" "}
                  <span className="text-yellow-600">({model.slug})</span>
                </p>
                {usageCount !== null && (
                  <p className="mt-2 font-semibold">
                    📊 Impact: {usageCount} block{usageCount !== 1 ? "s" : ""} currently use this model
                  </p>
                )}
                <p className="mt-2">
                  All workflows currently using this model will be automatically
                  updated to use the replacement model you choose below.
                </p>
              </div>
            </div>
          </div>

          <form action={handleDelete} className="space-y-4">
            <input type="hidden" name="model_id" value={model.id} />
            <input
              type="hidden"
              name="replacement_model_slug"
              value={selectedReplacement}
            />

            <label className="text-sm font-medium">
              <span className="block mb-2">
                Select Replacement Model <span className="text-red-500">*</span>
              </span>
              <select
                required
                value={selectedReplacement}
                onChange={(e) => setSelectedReplacement(e.target.value)}
                className="w-full rounded border border-input bg-background p-2 text-sm"
              >
                <option value="">-- Choose a replacement model --</option>
                {replacementOptions.map((m) => (
                  <option key={m.id} value={m.slug}>
                    {m.display_name} ({m.slug})
                  </option>
                ))}
              </select>
              {replacementOptions.length === 0 && (
                <p className="mt-2 text-xs text-red-600">
                  No replacement models available. You must have at least one
                  other enabled model before deleting this one.
                </p>
              )}
            </label>

            {error && (
              <div className="rounded-lg border border-red-200 bg-red-50 p-3 text-sm text-red-800">
                {error}
              </div>
            )}

            <Dialog.Footer>
              <Button
                variant="ghost"
                size="small"
                onClick={() => {
                  setOpen(false);
                  setSelectedReplacement("");
                  setError(null);
                }}
                disabled={isDeleting}
              >
                Cancel
              </Button>
              <button
                type="submit"
                disabled={
                  !selectedReplacement ||
                  isDeleting ||
                  replacementOptions.length === 0
                }
                className="inline-flex items-center rounded bg-red-600 px-4 py-2 text-sm font-semibold text-white hover:bg-red-700 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {isDeleting ? "Deleting..." : "Delete and Migrate"}
              </button>
            </Dialog.Footer>
          </form>
        </div>
      </Dialog.Content>
    </Dialog>
  );
}
