import type { LlmProvider } from "@/lib/autogpt-server-api/types";
import { createLlmModelAction } from "../actions";

export function AddModelForm({ providers }: { providers: LlmProvider[] }) {
  return (
    <form
      action={createLlmModelAction}
      className="space-y-8 rounded-lg border border-border bg-card p-8 shadow-sm"
    >
      <div className="space-y-2">
        <h2 className="text-2xl font-semibold tracking-tight">Add Model</h2>
        <p className="text-sm text-muted-foreground">
          Register a new model slug, metadata, and pricing.
        </p>
      </div>

      <div className="space-y-8">
        <div className="space-y-5">
          <div className="space-y-1">
            <h3 className="text-base font-semibold text-foreground">Basic Information</h3>
            <p className="text-xs text-muted-foreground">Core model details</p>
          </div>
          <div className="grid gap-5 md:grid-cols-2">
            <label className="space-y-2.5">
              <span className="text-sm font-medium text-foreground">Model Slug</span>
              <input
                required
                name="slug"
                className="w-full rounded-md border border-input bg-background px-4 py-2.5 text-sm transition-colors focus:border-primary focus:outline-none focus:ring-2 focus:ring-primary/20"
                placeholder="gpt-4.1-mini-2025-04-14"
              />
            </label>
            <label className="space-y-2.5">
              <span className="text-sm font-medium text-foreground">Display Name</span>
              <input
                required
                name="display_name"
                className="w-full rounded-md border border-input bg-background px-4 py-2.5 text-sm transition-colors focus:border-primary focus:outline-none focus:ring-2 focus:ring-primary/20"
                placeholder="GPT 4.1 Mini"
              />
            </label>
          </div>
          <label className="space-y-2.5">
            <span className="text-sm font-medium text-foreground">Description</span>
            <textarea
              name="description"
              rows={3}
              className="w-full rounded-md border border-input bg-background px-4 py-2.5 text-sm transition-colors focus:border-primary focus:outline-none focus:ring-2 focus:ring-primary/20"
              placeholder="Optional description..."
            />
          </label>
        </div>

        <div className="space-y-5 border-t border-border pt-6">
          <div className="space-y-1">
            <h3 className="text-base font-semibold text-foreground">Model Configuration</h3>
            <p className="text-xs text-muted-foreground">Model capabilities and limits</p>
          </div>
          <div className="grid gap-5 md:grid-cols-3">
            <label className="space-y-2.5">
              <span className="text-sm font-medium text-foreground">Provider</span>
              <select
                required
                name="provider_id"
                className="w-full rounded-md border border-input bg-background px-4 py-2.5 text-sm transition-colors focus:border-primary focus:outline-none focus:ring-2 focus:ring-primary/20"
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
            </label>
            <label className="space-y-2.5">
              <span className="text-sm font-medium text-foreground">Context Window</span>
              <input
                required
                type="number"
                name="context_window"
                className="w-full rounded-md border border-input bg-background px-4 py-2.5 text-sm transition-colors focus:border-primary focus:outline-none focus:ring-2 focus:ring-primary/20"
                placeholder="128000"
                min={1}
              />
            </label>
            <label className="space-y-2.5">
              <span className="text-sm font-medium text-foreground">Max Output Tokens</span>
              <input
                type="number"
                name="max_output_tokens"
                className="w-full rounded-md border border-input bg-background px-4 py-2.5 text-sm transition-colors focus:border-primary focus:outline-none focus:ring-2 focus:ring-primary/20"
                placeholder="16384"
                min={1}
              />
            </label>
          </div>
        </div>

        <div className="space-y-5 border-t border-border pt-6">
          <div className="space-y-1">
            <h3 className="text-base font-semibold text-foreground">Pricing & Credentials</h3>
            <p className="text-xs text-muted-foreground">Cost and credential configuration</p>
          </div>
          <div className="grid gap-5 md:grid-cols-2 lg:grid-cols-4">
            <label className="space-y-2.5">
              <span className="text-sm font-medium text-foreground">Credit Cost</span>
              <input
                required
                type="number"
                name="credit_cost"
                className="w-full rounded-md border border-input bg-background px-4 py-2.5 text-sm transition-colors focus:border-primary focus:outline-none focus:ring-2 focus:ring-primary/20"
                placeholder="5"
                min={0}
              />
            </label>
            <label className="space-y-2.5">
              <span className="text-sm font-medium text-foreground">Credential Provider</span>
              <input
                required
                name="credential_provider"
                className="w-full rounded-md border border-input bg-background px-4 py-2.5 text-sm transition-colors focus:border-primary focus:outline-none focus:ring-2 focus:ring-primary/20"
                placeholder="openai"
              />
            </label>
            <label className="space-y-2.5">
              <span className="text-sm font-medium text-foreground">Credential ID</span>
              <input
                name="credential_id"
                className="w-full rounded-md border border-input bg-background px-4 py-2.5 text-sm transition-colors focus:border-primary focus:outline-none focus:ring-2 focus:ring-primary/20"
                placeholder="cred-id"
              />
            </label>
            <label className="space-y-2.5">
              <span className="text-sm font-medium text-foreground">Credential Type</span>
              <input
                name="credential_type"
                defaultValue="api_key"
                className="w-full rounded-md border border-input bg-background px-4 py-2.5 text-sm transition-colors focus:border-primary focus:outline-none focus:ring-2 focus:ring-primary/20"
              />
            </label>
          </div>
          <p className="mt-3 text-xs text-muted-foreground">
            Credit cost is always in platform credits.
          </p>
        </div>

        <div className="space-y-3 border-t border-border pt-6">
          <label className="flex items-center gap-3 text-sm font-medium">
            <input type="hidden" name="is_enabled" value="off" />
            <input
              type="checkbox"
              name="is_enabled"
              defaultChecked
              className="h-4 w-4 rounded border-input"
            />
            Enabled by default
          </label>
        </div>
      </div>

      <div className="flex justify-end border-t border-border pt-6">
        <button
          type="submit"
          className="inline-flex items-center rounded-md bg-primary px-8 py-3 text-sm font-semibold text-primary-foreground shadow-sm transition-colors hover:bg-primary/90"
        >
          Save Model
        </button>
      </div>
    </form>
  );
}

