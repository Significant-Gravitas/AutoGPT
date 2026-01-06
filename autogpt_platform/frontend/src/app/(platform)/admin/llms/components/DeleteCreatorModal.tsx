"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { Button } from "@/components/atoms/Button/Button";
import type { LlmModelCreator } from "@/app/api/__generated__/models/llmModelCreator";
import { deleteLlmCreatorAction } from "../actions";

export function DeleteCreatorModal({ creator }: { creator: LlmModelCreator }) {
  const [open, setOpen] = useState(false);
  const [isDeleting, setIsDeleting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const router = useRouter();

  async function handleDelete(formData: FormData) {
    setIsDeleting(true);
    setError(null);
    try {
      await deleteLlmCreatorAction(formData);
      setOpen(false);
      router.refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to delete creator");
    } finally {
      setIsDeleting(false);
    }
  }

  return (
    <Dialog
      title="Delete Creator"
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
          <div className="rounded-lg border border-amber-500/30 bg-amber-500/10 p-4 dark:border-amber-400/30 dark:bg-amber-400/10">
            <div className="flex items-start gap-3">
              <div className="flex-shrink-0 text-amber-600 dark:text-amber-400">
                ⚠️
              </div>
              <div className="text-sm text-foreground">
                <p className="font-semibold">You are about to delete:</p>
                <p className="mt-1">
                  <span className="font-medium">{creator.display_name}</span>{" "}
                  <span className="text-muted-foreground">
                    ({creator.name})
                  </span>
                </p>
                <p className="mt-2 text-muted-foreground">
                  Models using this creator will have their creator field
                  cleared. This is safe and won&apos;t affect model
                  functionality.
                </p>
              </div>
            </div>
          </div>

          <form action={handleDelete} className="space-y-4">
            <input type="hidden" name="creator_id" value={creator.id} />

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
                disabled={isDeleting}
                className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
              >
                {isDeleting ? "Deleting..." : "Delete Creator"}
              </Button>
            </Dialog.Footer>
          </form>
        </div>
      </Dialog.Content>
    </Dialog>
  );
}
