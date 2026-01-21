"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { Button } from "@/components/atoms/Button/Button";
import type { LlmModel } from "@/app/api/__generated__/models/llmModel";
import { deleteLlmModelAction, fetchLlmModelUsage } from "../actions";

export function DeleteModelModal({
  model,
  availableModels,
}: {
  model: LlmModel;
  availableModels: LlmModel[];
}) {
  const router = useRouter();
  const [open, setOpen] = useState(false);
  const [selectedReplacement, setSelectedReplacement] = useState<string>("");
  const [isDeleting, setIsDeleting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [usageCount, setUsageCount] = useState<number | null>(null);
  const [usageLoading, setUsageLoading] = useState(false);
  const [usageError, setUsageError] = useState<string | null>(null);

  // Filter out the current model and disabled models from replacement options
  const replacementOptions = availableModels.filter(
    (m) => m.id !== model.id && m.is_enabled,
  );

  // Check if migration is required (has blocks using this model)
  const requiresMigration = usageCount !== null && usageCount > 0;

  async function fetchUsage() {
    setUsageLoading(true);
    setUsageError(null);
    try {
      const usage = await fetchLlmModelUsage(model.id);
      setUsageCount(usage.node_count);
    } catch (err) {
      console.error("Failed to fetch model usage:", err);
      setUsageError("Failed to load usage count");
      setUsageCount(null);
    } finally {
      setUsageLoading(false);
    }
  }

  async function handleDelete(formData: FormData) {
    setIsDeleting(true);
    setError(null);
    try {
      await deleteLlmModelAction(formData);
      setOpen(false);
      router.refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to delete model");
    } finally {
      setIsDeleting(false);
    }
  }

  // Determine if delete button should be enabled
  const canDelete =
    !isDeleting &&
    !usageLoading &&
    usageCount !== null &&
    (requiresMigration
      ? selectedReplacement && replacementOptions.length > 0
      : true);

  return (
    <Dialog
      title="Delete Model"
      controlled={{
        isOpen: open,
        set: async (isOpen) => {
          setOpen(isOpen);
          if (isOpen) {
            setUsageCount(null);
            setUsageError(null);
            setError(null);
            setSelectedReplacement("");
            await fetchUsage();
          }
        },
      }}
      styling={{ maxWidth: "600px" }}
    >
      <Dialog.Trigger>
        <Button
          type="button"
          variant="outline"
          size="small"
          className="min-w-0 text-destructive hover:bg-destructive/10"
        >
          Delete
        </Button>
      </Dialog.Trigger>
      <Dialog.Content>
        <div className="mb-4 text-sm text-muted-foreground">
          {requiresMigration
            ? "This action cannot be undone. All workflows using this model will be migrated to the replacement model you select."
            : "This action cannot be undone."}
        </div>

        <div className="space-y-4">
          <div className="rounded-lg border border-amber-500/30 bg-amber-500/10 p-4 dark:border-amber-400/30 dark:bg-amber-400/10">
            <div className="flex items-start gap-3">
              <div className="flex-shrink-0 text-amber-600 dark:text-amber-400">
                ⚠️
              </div>
              <div className="text-sm text-foreground">
                <p className="font-semibold">You are about to delete:</p>
                <p className="mt-1">
                  <span className="font-medium">{model.display_name}</span>{" "}
                  <span className="text-muted-foreground">({model.slug})</span>
                </p>
                {usageLoading && (
                  <p className="mt-2 text-muted-foreground">
                    Loading usage count...
                  </p>
                )}
                {usageError && (
                  <p className="mt-2 text-destructive">{usageError}</p>
                )}
                {!usageLoading && !usageError && usageCount !== null && (
                  <p className="mt-2 font-semibold">
                    Impact: {usageCount} block{usageCount !== 1 ? "s" : ""}{" "}
                    currently use this model
                  </p>
                )}
                {requiresMigration && (
                  <p className="mt-2 text-muted-foreground">
                    All workflows currently using this model will be
                    automatically updated to use the replacement model you
                    choose below.
                  </p>
                )}
                {!usageLoading && usageCount === 0 && (
                  <p className="mt-2 text-muted-foreground">
                    No workflows are using this model. It can be safely deleted.
                  </p>
                )}
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

            {requiresMigration && (
              <label className="text-sm font-medium">
                <span className="mb-2 block">
                  Select Replacement Model{" "}
                  <span className="text-destructive">*</span>
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
                  <p className="mt-2 text-xs text-destructive">
                    No replacement models available. You must have at least one
                    other enabled model before deleting this one.
                  </p>
                )}
              </label>
            )}

            {error && (
              <div className="rounded-lg border border-destructive/30 bg-destructive/10 p-3 text-sm text-destructive">
                {error}
              </div>
            )}

            <Dialog.Footer>
              <Button
                variant="ghost"
                size="small"
                type="button"
                onClick={() => {
                  setOpen(false);
                  setSelectedReplacement("");
                  setError(null);
                }}
                disabled={isDeleting}
              >
                Cancel
              </Button>
              <Button
                type="submit"
                variant="primary"
                size="small"
                disabled={!canDelete}
                className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
              >
                {isDeleting
                  ? "Deleting..."
                  : requiresMigration
                    ? "Delete and Migrate"
                    : "Delete"}
              </Button>
            </Dialog.Footer>
          </form>
        </div>
      </Dialog.Content>
    </Dialog>
  );
}
