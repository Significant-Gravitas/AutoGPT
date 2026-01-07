"use client";

import { useState } from "react";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { Button } from "@/components/atoms/Button/Button";
import { createLlmCreatorAction } from "../actions";
import { useRouter } from "next/navigation";

export function AddCreatorModal() {
  const [open, setOpen] = useState(false);
  const router = useRouter();

  return (
    <Dialog
      title="Add Creator"
      controlled={{ isOpen: open, set: setOpen }}
      styling={{ maxWidth: "512px" }}
    >
      <Dialog.Trigger>
        <Button variant="primary" size="small">
          Add Creator
        </Button>
      </Dialog.Trigger>
      <Dialog.Content>
        <div className="mb-4 text-sm text-muted-foreground">
          Add a new model creator (the organization that made/trained the
          model).
        </div>

        <form
          action={async (formData) => {
            await createLlmCreatorAction(formData);
            setOpen(false);
            router.refresh();
          }}
          className="space-y-4"
        >
          <div className="grid gap-4 sm:grid-cols-2">
            <div className="space-y-2">
              <label
                htmlFor="name"
                className="text-sm font-medium text-foreground"
              >
                Name (slug) <span className="text-destructive">*</span>
              </label>
              <input
                id="name"
                required
                name="name"
                className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm transition-colors placeholder:text-muted-foreground focus:border-primary focus:outline-none focus:ring-2 focus:ring-primary/20"
                placeholder="openai"
              />
              <p className="text-xs text-muted-foreground">
                Lowercase identifier (e.g., openai, meta, anthropic)
              </p>
            </div>
            <div className="space-y-2">
              <label
                htmlFor="display_name"
                className="text-sm font-medium text-foreground"
              >
                Display Name <span className="text-destructive">*</span>
              </label>
              <input
                id="display_name"
                required
                name="display_name"
                className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm transition-colors placeholder:text-muted-foreground focus:border-primary focus:outline-none focus:ring-2 focus:ring-primary/20"
                placeholder="OpenAI"
              />
            </div>
          </div>

          <div className="space-y-2">
            <label
              htmlFor="description"
              className="text-sm font-medium text-foreground"
            >
              Description
            </label>
            <textarea
              id="description"
              name="description"
              rows={2}
              className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm transition-colors placeholder:text-muted-foreground focus:border-primary focus:outline-none focus:ring-2 focus:ring-primary/20"
              placeholder="Creator of GPT models..."
            />
          </div>

          <div className="space-y-2">
            <label
              htmlFor="website_url"
              className="text-sm font-medium text-foreground"
            >
              Website URL
            </label>
            <input
              id="website_url"
              name="website_url"
              type="url"
              className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm transition-colors placeholder:text-muted-foreground focus:border-primary focus:outline-none focus:ring-2 focus:ring-primary/20"
              placeholder="https://openai.com"
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
              Add Creator
            </Button>
          </Dialog.Footer>
        </form>
      </Dialog.Content>
    </Dialog>
  );
}
