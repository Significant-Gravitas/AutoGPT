"use client";

import { useState } from "react";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { Button } from "@/components/atoms/Button/Button";
import { createLlmProviderAction } from "../actions";
import { useRouter } from "next/navigation";

export function AddProviderModal() {
  const [open, setOpen] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const router = useRouter();

  async function handleSubmit(formData: FormData) {
    setIsSubmitting(true);
    setError(null);
    try {
      await createLlmProviderAction(formData);
      setOpen(false);
      router.refresh();
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Failed to create provider",
      );
    } finally {
      setIsSubmitting(false);
    }
  }

  return (
    <Dialog
      title="Add Provider"
      controlled={{ isOpen: open, set: setOpen }}
      styling={{ maxWidth: "768px", maxHeight: "90vh", overflowY: "auto" }}
    >
      <Dialog.Trigger>
        <Button variant="primary" size="small">
          Add Provider
        </Button>
      </Dialog.Trigger>
      <Dialog.Content>
        <div className="mb-4 text-sm text-muted-foreground">
          Define a new upstream provider and default credential information.
        </div>

        {/* Setup Instructions */}
        <div className="mb-6 rounded-lg border border-primary/30 bg-primary/5 p-4">
          <div className="space-y-2">
            <h4 className="text-sm font-semibold text-foreground">
              Before Adding a Provider
            </h4>
            <p className="text-xs text-muted-foreground">
              To use a new provider, you must first configure its credentials in
              the backend:
            </p>
            <ol className="list-inside list-decimal space-y-1 text-xs text-muted-foreground">
              <li>
                Add the credential to{" "}
                <code className="rounded bg-muted px-1 py-0.5 font-mono">
                  backend/integrations/credentials_store.py
                </code>{" "}
                with a UUID, provider name, and settings secret reference
              </li>
              <li>
                Add it to the{" "}
                <code className="rounded bg-muted px-1 py-0.5 font-mono">
                  PROVIDER_CREDENTIALS
                </code>{" "}
                dictionary in{" "}
                <code className="rounded bg-muted px-1 py-0.5 font-mono">
                  backend/data/block_cost_config.py
                </code>
              </li>
              <li>
                Use the <strong>same provider name</strong> in the
                &quot;Credential Provider&quot; field below that matches the key
                in{" "}
                <code className="rounded bg-muted px-1 py-0.5 font-mono">
                  PROVIDER_CREDENTIALS
                </code>
              </li>
            </ol>
          </div>
        </div>

        <form action={handleSubmit} className="space-y-6">
          {/* Basic Information */}
          <div className="space-y-4">
            <div className="space-y-1">
              <h3 className="text-sm font-semibold text-foreground">
                Basic Information
              </h3>
              <p className="text-xs text-muted-foreground">
                Core provider details
              </p>
            </div>
            <div className="grid gap-4 sm:grid-cols-2">
              <div className="space-y-2">
                <label
                  htmlFor="name"
                  className="text-sm font-medium text-foreground"
                >
                  Provider Slug <span className="text-destructive">*</span>
                </label>
                <input
                  id="name"
                  required
                  name="name"
                  className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm transition-colors placeholder:text-muted-foreground focus:border-primary focus:outline-none focus:ring-2 focus:ring-primary/20"
                  placeholder="e.g. openai"
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
                rows={3}
                className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm transition-colors placeholder:text-muted-foreground focus:border-primary focus:outline-none focus:ring-2 focus:ring-primary/20"
                placeholder="Optional description..."
              />
            </div>
          </div>

          {/* Default Credentials */}
          <div className="space-y-4 border-t border-border pt-6">
            <div className="space-y-1">
              <h3 className="text-sm font-semibold text-foreground">
                Default Credentials
              </h3>
              <p className="text-xs text-muted-foreground">
                Credential provider name that matches the key in{" "}
                <code className="rounded bg-muted px-1 py-0.5 font-mono text-xs">
                  PROVIDER_CREDENTIALS
                </code>
              </p>
            </div>
            <div className="space-y-2">
              <label
                htmlFor="default_credential_provider"
                className="text-sm font-medium text-foreground"
              >
                Credential Provider <span className="text-destructive">*</span>
              </label>
              <input
                id="default_credential_provider"
                name="default_credential_provider"
                required
                className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm transition-colors placeholder:text-muted-foreground focus:border-primary focus:outline-none focus:ring-2 focus:ring-primary/20"
                placeholder="openai"
              />
              <p className="text-xs text-muted-foreground">
                <strong>Important:</strong> This must exactly match the key in
                the{" "}
                <code className="rounded bg-muted px-1 py-0.5 font-mono text-xs">
                  PROVIDER_CREDENTIALS
                </code>{" "}
                dictionary in{" "}
                <code className="rounded bg-muted px-1 py-0.5 font-mono text-xs">
                  block_cost_config.py
                </code>
                . Common values: &quot;openai&quot;, &quot;anthropic&quot;,
                &quot;groq&quot;, &quot;open_router&quot;, etc.
              </p>
            </div>
          </div>

          {/* Capabilities */}
          <div className="space-y-4 border-t border-border pt-6">
            <div className="space-y-1">
              <h3 className="text-sm font-semibold text-foreground">
                Capabilities
              </h3>
              <p className="text-xs text-muted-foreground">
                Provider feature flags
              </p>
            </div>
            <div className="grid gap-3 sm:grid-cols-2">
              {[
                { name: "supports_tools", label: "Supports tools" },
                { name: "supports_json_output", label: "Supports JSON output" },
                { name: "supports_reasoning", label: "Supports reasoning" },
                {
                  name: "supports_parallel_tool",
                  label: "Supports parallel tool calls",
                },
              ].map(({ name, label }) => (
                <div
                  key={name}
                  className="flex items-center gap-3 rounded-md border border-border bg-muted/30 px-4 py-3 transition-colors hover:bg-muted/50"
                >
                  <input type="hidden" name={name} value="off" />
                  <input
                    id={name}
                    type="checkbox"
                    name={name}
                    defaultChecked={
                      name !== "supports_reasoning" &&
                      name !== "supports_parallel_tool"
                    }
                    className="h-4 w-4 rounded border-input"
                  />
                  <label
                    htmlFor={name}
                    className="text-sm font-medium text-foreground"
                  >
                    {label}
                  </label>
                </div>
              ))}
            </div>
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
              {isSubmitting ? "Creating..." : "Save Provider"}
            </Button>
          </Dialog.Footer>
        </form>
      </Dialog.Content>
    </Dialog>
  );
}
