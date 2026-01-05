"use client";

import { useState } from "react";
import type { LlmModelCreator } from "@/lib/autogpt-server-api/types";
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
import {
  deleteLlmCreatorAction,
  updateLlmCreatorAction,
} from "../actions";
import { useRouter } from "next/navigation";

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
                    {new URL(creator.website_url).hostname}
                  </a>
                ) : (
                  <span className="text-muted-foreground">—</span>
                )}
              </TableCell>
              <TableCell>
                <div className="flex items-center justify-end gap-2">
                  <EditCreatorModal creator={creator} />
                  <DeleteCreatorButton creatorId={creator.id} />
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
  const router = useRouter();

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
        <form
          action={async (formData) => {
            await updateLlmCreatorAction(formData);
            setOpen(false);
            router.refresh();
          }}
          className="space-y-4"
        >
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

          <Dialog.Footer>
            <Button
              variant="ghost"
              size="small"
              onClick={() => setOpen(false)}
              type="button"
            >
              Cancel
            </Button>
            <Button variant="primary" size="small" type="submit">
              Update
            </Button>
          </Dialog.Footer>
        </form>
      </Dialog.Content>
    </Dialog>
  );
}

function DeleteCreatorButton({ creatorId }: { creatorId: string }) {
  const router = useRouter();

  return (
    <form
      action={async (formData) => {
        if (
          confirm(
            "Delete this creator? Models using this creator will have their creator set to none."
          )
        ) {
          await deleteLlmCreatorAction(formData);
          router.refresh();
        }
      }}
    >
      <input type="hidden" name="creator_id" value={creatorId} />
      <Button
        type="submit"
        variant="outline"
        size="small"
        className="min-w-0 text-destructive hover:bg-destructive/10"
      >
        Delete
      </Button>
    </form>
  );
}
