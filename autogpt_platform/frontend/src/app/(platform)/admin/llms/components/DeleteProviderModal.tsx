"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { Button } from "@/components/atoms/Button/Button";
import type { LlmProvider } from "@/app/api/__generated__/models/llmProvider";
import { deleteLlmProviderAction } from "../actions";

export function DeleteProviderModal({ provider }: { provider: LlmProvider }) {
  const [open, setOpen] = useState(false);
  const [isDeleting, setIsDeleting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const router = useRouter();

  const modelCount = provider.models?.length ?? 0;
  const hasModels = modelCount > 0;

  async function handleDelete(formData: FormData) {
    setIsDeleting(true);
    setError(null);
    try {
      await deleteLlmProviderAction(formData);
      setOpen(false);
      router.refresh();
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Failed to delete provider",
      );
    } finally {
      setIsDeleting(false);
    }
  }

  return (
    <Dialog
      title="Delete Provider"
      controlled={{ isOpen: open, set: setOpen }}
      styling={{ maxWidth: "480px" }}
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
        <div className="space-y-4">
          <div
            className={`rounded-lg border p-4 ${
              hasModels
                ? "border-destructive/30 bg-destructive/10"
                : "border-amber-500/30 bg-amber-500/10 dark:border-amber-400/30 dark:bg-amber-400/10"
            }`}
          >
            <div className="flex items-start gap-3">
              <div
                className={`flex-shrink-0 ${
                  hasModels
                    ? "text-destructive"
                    : "text-amber-600 dark:text-amber-400"
                }`}
              >
                {hasModels ? "🚫" : "⚠️"}
              </div>
              <div className="text-sm text-foreground">
                <p className="font-semibold">You are about to delete:</p>
                <p className="mt-1">
                  <span className="font-medium">{provider.display_name}</span>{" "}
                  <span className="text-muted-foreground">
                    ({provider.name})
                  </span>
                </p>
                {hasModels ? (
                  <p className="mt-2 text-destructive">
                    This provider has {modelCount} model(s). You must delete all
                    models before you can delete this provider.
                  </p>
                ) : (
                  <p className="mt-2 text-muted-foreground">
                    This provider has no models and can be safely deleted.
                  </p>
                )}
              </div>
            </div>
          </div>

          <form action={handleDelete} className="space-y-4">
            <input type="hidden" name="provider_id" value={provider.id} />

            {error && (
              <div className="rounded-lg border border-destructive/30 bg-destructive/10 p-3 text-sm text-destructive">
                {error}
              </div>
            )}

            <Dialog.Footer>
              <Button
                variant="ghost"
                size="small"
                onClick={() => {
                  setOpen(false);
                  setError(null);
                }}
                disabled={isDeleting}
                type="button"
              >
                Cancel
              </Button>
              <Button
                type="submit"
                variant="primary"
                size="small"
                disabled={isDeleting || hasModels}
                className="bg-destructive text-destructive-foreground hover:bg-destructive/90 disabled:opacity-50"
              >
                {isDeleting ? "Deleting..." : "Delete Provider"}
              </Button>
            </Dialog.Footer>
          </form>
        </div>
      </Dialog.Content>
    </Dialog>
  );
}
