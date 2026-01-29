"use client";

import { useState } from "react";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { Button } from "@/components/atoms/Button/Button";
import { updateLlmProviderAction } from "../actions";
import { useRouter } from "next/navigation";
import type { LlmProvider } from "@/app/api/__generated__/models/llmProvider";

export function EditProviderModal({ provider }: { provider: LlmProvider }) {
  const [open, setOpen] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const router = useRouter();

  async function handleSubmit(formData: FormData) {
    setIsSubmitting(true);
    setError(null);
    try {
      await updateLlmProviderAction(formData);
      setOpen(false);
      router.refresh();
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Failed to update provider",
      );
    } finally {
      setIsSubmitting(false);
    }
  }

  return (
    <Dialog
      title="Edit Provider"
      controlled={{ isOpen: open, set: setOpen }}
      styling={{ maxWidth: "768px", maxHeight: "90vh", overflowY: "auto" }}
    >
      <Dialog.Trigger>
        <Button variant="outline" size="small">
          Edit
        </Button>
      </Dialog.Trigger>
      <Dialog.Content>
        <div className="mb-4 text-sm text-muted-foreground">
          Update provider configuration and capabilities.
        </div>

        <form action={handleSubmit} className="space-y-6">
          <input type="hidden" name="provider_id" value={provider.id} />

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
                  defaultValue={provider.name}
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
                  defaultValue={provider.display_name}
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
                defaultValue={provider.description ?? ""}
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
            <div className="grid gap-4 sm:grid-cols-2">
              <div className="space-y-2">
                <label
                  htmlFor="default_credential_provider"
                  className="text-sm font-medium text-foreground"
                >
                  Credential Provider
                </label>
                <input
                  id="default_credential_provider"
                  name="default_credential_provider"
                  defaultValue={provider.default_credential_provider ?? ""}
                  className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm transition-colors placeholder:text-muted-foreground focus:border-primary focus:outline-none focus:ring-2 focus:ring-primary/20"
                  placeholder="openai"
                />
              </div>
              <div className="space-y-2">
                <label
                  htmlFor="default_credential_id"
                  className="text-sm font-medium text-foreground"
                >
                  Credential ID
                </label>
                <input
                  id="default_credential_id"
                  name="default_credential_id"
                  defaultValue={provider.default_credential_id ?? ""}
                  className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm transition-colors placeholder:text-muted-foreground focus:border-primary focus:outline-none focus:ring-2 focus:ring-primary/20"
                  placeholder="Optional credential ID"
                />
              </div>
            </div>
            <div className="space-y-2">
              <label
                htmlFor="default_credential_type"
                className="text-sm font-medium text-foreground"
              >
                Credential Type
              </label>
              <input
                id="default_credential_type"
                name="default_credential_type"
                defaultValue={provider.default_credential_type ?? "api_key"}
                className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm transition-colors placeholder:text-muted-foreground focus:border-primary focus:outline-none focus:ring-2 focus:ring-primary/20"
                placeholder="api_key"
              />
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
                {
                  name: "supports_tools",
                  label: "Supports tools",
                  checked: provider.supports_tools,
                },
                {
                  name: "supports_json_output",
                  label: "Supports JSON output",
                  checked: provider.supports_json_output,
                },
                {
                  name: "supports_reasoning",
                  label: "Supports reasoning",
                  checked: provider.supports_reasoning,
                },
                {
                  name: "supports_parallel_tool",
                  label: "Supports parallel tool calls",
                  checked: provider.supports_parallel_tool,
                },
              ].map(({ name, label, checked }) => (
                <div
                  key={name}
                  className="flex items-center gap-3 rounded-md border border-border bg-muted/30 px-4 py-3 transition-colors hover:bg-muted/50"
                >
                  <input type="hidden" name={name} value="off" />
                  <input
                    id={name}
                    type="checkbox"
                    name={name}
                    defaultChecked={checked}
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
              {isSubmitting ? "Saving..." : "Save Changes"}
            </Button>
          </Dialog.Footer>
        </form>
      </Dialog.Content>
    </Dialog>
  );
}
