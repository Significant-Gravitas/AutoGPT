"use client";

import { useState } from "react";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { Button } from "@/components/atoms/Button/Button";
import type { LlmProvider } from "@/lib/autogpt-server-api/types";
import { createLlmModelAction } from "../actions";
import { useRouter } from "next/navigation";

interface Props {
  providers: LlmProvider[];
}

export function AddModelModal({ providers }: Props) {
  const [open, setOpen] = useState(false);
  const router = useRouter();

  return (
    <Dialog
      title="Add Model"
      controlled={{ isOpen: open, set: setOpen }}
      styling={{ maxWidth: "768px", maxHeight: "90vh", overflowY: "auto" }}
    >
      <Dialog.Trigger>
        <Button variant="primary" size="small">
          Add Model
        </Button>
      </Dialog.Trigger>
      <Dialog.Content>
        <div className="mb-4 text-sm text-muted-foreground">
          Register a new model slug, metadata, and pricing.
        </div>

        <form
          action={async (formData) => {
            await createLlmModelAction(formData);
            setOpen(false);
            router.refresh();
          }}
          className="space-y-6"
        >
          {/* Basic Information */}
          <div className="space-y-4">
            <div className="space-y-1">
              <h3 className="text-sm font-semibold text-foreground">
                Basic Information
              </h3>
              <p className="text-xs text-muted-foreground">Core model details</p>
            </div>
            <div className="grid gap-4 sm:grid-cols-2">
              <div className="space-y-2">
                <label
                  htmlFor="slug"
                  className="text-sm font-medium text-foreground"
                >
                  Model Slug <span className="text-destructive">*</span>
                </label>
                <input
                  id="slug"
                  required
                  name="slug"
                  className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm transition-colors placeholder:text-muted-foreground focus:border-primary focus:outline-none focus:ring-2 focus:ring-primary/20"
                  placeholder="gpt-4.1-mini-2025-04-14"
                />
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
                  placeholder="GPT 4.1 Mini"
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
                rows={3}
                className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm transition-colors placeholder:text-muted-foreground focus:border-primary focus:outline-none focus:ring-2 focus:ring-primary/20"
                placeholder="Optional description..."
              />
            </div>
          </div>

          {/* Model Configuration */}
          <div className="space-y-4 border-t border-border pt-6">
            <div className="space-y-1">
              <h3 className="text-sm font-semibold text-foreground">
                Model Configuration
              </h3>
              <p className="text-xs text-muted-foreground">
                Model capabilities and limits
              </p>
            </div>
            <div className="grid gap-4 sm:grid-cols-3">
              <div className="space-y-2">
                <label
                  htmlFor="provider_id"
                  className="text-sm font-medium text-foreground"
                >
                  Provider <span className="text-destructive">*</span>
                </label>
                <select
                  id="provider_id"
                  required
                  name="provider_id"
                  className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm transition-colors focus:border-primary focus:outline-none focus:ring-2 focus:ring-primary/20"
                  defaultValue=""
                >
                  <option value="" disabled>
                    Select provider
                  </option>
                  {providers.map((provider) => (
                    <option key={provider.id} value={provider.id}>
                      {provider.display_name} ({provider.name})
                    </option>
                  ))}
                </select>
              </div>
              <div className="space-y-2">
                <label
                  htmlFor="context_window"
                  className="text-sm font-medium text-foreground"
                >
                  Context Window <span className="text-destructive">*</span>
                </label>
                <input
                  id="context_window"
                  required
                  type="number"
                  name="context_window"
                  className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm transition-colors placeholder:text-muted-foreground focus:border-primary focus:outline-none focus:ring-2 focus:ring-primary/20"
                  placeholder="128000"
                  min={1}
                />
              </div>
              <div className="space-y-2">
                <label
                  htmlFor="max_output_tokens"
                  className="text-sm font-medium text-foreground"
                >
                  Max Output Tokens
                </label>
                <input
                  id="max_output_tokens"
                  type="number"
                  name="max_output_tokens"
                  className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm transition-colors placeholder:text-muted-foreground focus:border-primary focus:outline-none focus:ring-2 focus:ring-primary/20"
                  placeholder="16384"
                  min={1}
                />
              </div>
            </div>
          </div>

          {/* Pricing */}
          <div className="space-y-4 border-t border-border pt-6">
            <div className="space-y-1">
              <h3 className="text-sm font-semibold text-foreground">Pricing</h3>
              <p className="text-xs text-muted-foreground">
                Credit cost per run (credentials are managed via the provider)
              </p>
            </div>
            <div className="grid gap-4 sm:grid-cols-1">
              <div className="space-y-2">
                <label
                  htmlFor="credit_cost"
                  className="text-sm font-medium text-foreground"
                >
                  Credit Cost <span className="text-destructive">*</span>
                </label>
                <input
                  id="credit_cost"
                  required
                  type="number"
                  name="credit_cost"
                  className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm transition-colors placeholder:text-muted-foreground focus:border-primary focus:outline-none focus:ring-2 focus:ring-primary/20"
                  placeholder="5"
                  min={0}
                />
              </div>
            </div>
            <p className="text-xs text-muted-foreground">
              Credit cost is always in platform credits. Credentials are inherited
              from the selected provider.
            </p>
          </div>

          {/* Enabled Toggle */}
          <div className="flex items-center gap-3 border-t border-border pt-6">
            <input type="hidden" name="is_enabled" value="off" />
            <input
              id="is_enabled"
              type="checkbox"
              name="is_enabled"
              defaultChecked
              className="h-4 w-4 rounded border-input"
            />
            <label
              htmlFor="is_enabled"
              className="text-sm font-medium text-foreground"
            >
              Enabled by default
            </label>
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
              Save Model
            </Button>
          </Dialog.Footer>
        </form>
      </Dialog.Content>
    </Dialog>
  );
}

