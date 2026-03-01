"use client";

import { useState } from "react";
import type { LlmModelCreator } from "@/app/api/__generated__/models/llmModelCreator";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/atoms/Table/Table";
import { Button } from "@/components/atoms/Button/Button";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { updateLlmCreatorAction } from "../actions";
import { useRouter } from "next/navigation";
import { DeleteCreatorModal } from "./DeleteCreatorModal";

export function CreatorsTable({ creators }: { creators: LlmModelCreator[] }) {
  if (!creators.length) {
    return (
      <div className="rounded-lg border border-dashed border-border p-6 text-center text-sm text-muted-foreground">
        No creators registered yet.
      </div>
    );
  }

  return (
    <div className="rounded-lg border">
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Creator</TableHead>
            <TableHead>Description</TableHead>
            <TableHead>Website</TableHead>
            <TableHead>Actions</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {creators.map((creator) => (
            <TableRow key={creator.id}>
              <TableCell>
                <div className="font-medium">{creator.display_name}</div>
                <div className="text-xs text-muted-foreground">
                  {creator.name}
                </div>
              </TableCell>
              <TableCell>
                <span className="text-sm text-muted-foreground">
                  {creator.description || "—"}
                </span>
              </TableCell>
              <TableCell>
                {creator.website_url ? (
                  <a
                    href={creator.website_url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-sm text-primary hover:underline"
                  >
                    {(() => {
                      try {
                        return new URL(creator.website_url).hostname;
                      } catch {
                        return creator.website_url;
                      }
                    })()}
                  </a>
                ) : (
                  <span className="text-muted-foreground">—</span>
                )}
              </TableCell>
              <TableCell>
                <div className="flex items-center justify-end gap-2">
                  <EditCreatorModal creator={creator} />
                  <DeleteCreatorModal creator={creator} />
                </div>
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  );
}

function EditCreatorModal({ creator }: { creator: LlmModelCreator }) {
  const [open, setOpen] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const router = useRouter();

  async function handleSubmit(formData: FormData) {
    setIsSubmitting(true);
    setError(null);
    try {
      await updateLlmCreatorAction(formData);
      setOpen(false);
      router.refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to update creator");
    } finally {
      setIsSubmitting(false);
    }
  }

  return (
    <Dialog
      title="Edit Creator"
      controlled={{ isOpen: open, set: setOpen }}
      styling={{ maxWidth: "512px" }}
    >
      <Dialog.Trigger>
        <Button variant="outline" size="small" className="min-w-0">
          Edit
        </Button>
      </Dialog.Trigger>
      <Dialog.Content>
        <form action={handleSubmit} className="space-y-4">
          <input type="hidden" name="creator_id" value={creator.id} />

          <div className="grid gap-4 sm:grid-cols-2">
            <div className="space-y-2">
              <label className="text-sm font-medium">Name (slug)</label>
              <input
                required
                name="name"
                defaultValue={creator.name}
                className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
              />
            </div>
            <div className="space-y-2">
              <label className="text-sm font-medium">Display Name</label>
              <input
                required
                name="display_name"
                defaultValue={creator.display_name}
                className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
              />
            </div>
          </div>

          <div className="space-y-2">
            <label className="text-sm font-medium">Description</label>
            <textarea
              name="description"
              rows={2}
              defaultValue={creator.description ?? ""}
              className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
            />
          </div>

          <div className="space-y-2">
            <label className="text-sm font-medium">Website URL</label>
            <input
              name="website_url"
              type="url"
              defaultValue={creator.website_url ?? ""}
              className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
            />
          </div>

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
                setError(null);
              }}
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
              {isSubmitting ? "Updating..." : "Update"}
            </Button>
          </Dialog.Footer>
        </form>
      </Dialog.Content>
    </Dialog>
  );
}
