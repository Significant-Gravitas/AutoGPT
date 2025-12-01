import { createLlmProviderAction } from "../actions";

export function AddProviderForm() {
  return (
    <form
      action={createLlmProviderAction}
      className="space-y-8 rounded-lg border border-border bg-card p-8 shadow-sm"
    >
      <div className="space-y-2">
        <h2 className="text-2xl font-semibold tracking-tight">Add Provider</h2>
        <p className="text-sm text-muted-foreground">
          Define a new upstream provider and default credential information.
        </p>
      </div>

      <div className="space-y-8">
        <div className="space-y-5">
          <div className="space-y-1">
            <h3 className="text-base font-semibold text-foreground">Basic Information</h3>
            <p className="text-xs text-muted-foreground">Core provider details</p>
          </div>
          <div className="grid gap-5 md:grid-cols-2">
            <label className="space-y-2.5">
              <span className="text-sm font-medium text-foreground">Provider Slug</span>
              <input
                required
                name="name"
                className="w-full rounded-md border border-input bg-background px-4 py-2.5 text-sm transition-colors focus:border-primary focus:outline-none focus:ring-2 focus:ring-primary/20"
                placeholder="e.g. openai"
              />
            </label>
            <label className="space-y-2.5">
              <span className="text-sm font-medium text-foreground">Display Name</span>
              <input
                required
                name="display_name"
                className="w-full rounded-md border border-input bg-background px-4 py-2.5 text-sm transition-colors focus:border-primary focus:outline-none focus:ring-2 focus:ring-primary/20"
                placeholder="OpenAI"
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
            <h3 className="text-base font-semibold text-foreground">Default Credentials</h3>
            <p className="text-xs text-muted-foreground">Default credential configuration</p>
          </div>
          <div className="grid gap-5 md:grid-cols-3">
            <label className="space-y-2.5">
              <span className="text-sm font-medium text-foreground">Credential Provider</span>
              <input
                name="default_credential_provider"
                className="w-full rounded-md border border-input bg-background px-4 py-2.5 text-sm transition-colors focus:border-primary focus:outline-none focus:ring-2 focus:ring-primary/20"
                placeholder="openai"
              />
            </label>
            <label className="space-y-2.5">
              <span className="text-sm font-medium text-foreground">Credential ID</span>
              <input
                name="default_credential_id"
                className="w-full rounded-md border border-input bg-background px-4 py-2.5 text-sm transition-colors focus:border-primary focus:outline-none focus:ring-2 focus:ring-primary/20"
                placeholder="cred-id"
              />
            </label>
            <label className="space-y-2.5">
              <span className="text-sm font-medium text-foreground">Credential Type</span>
              <input
                name="default_credential_type"
                defaultValue="api_key"
                className="w-full rounded-md border border-input bg-background px-4 py-2.5 text-sm transition-colors focus:border-primary focus:outline-none focus:ring-2 focus:ring-primary/20"
              />
            </label>
          </div>
        </div>

        <div className="space-y-5 border-t border-border pt-6">
          <div className="space-y-1">
            <h3 className="text-base font-semibold text-foreground">Capabilities</h3>
            <p className="text-xs text-muted-foreground">Provider feature flags</p>
          </div>
          <div className="grid gap-4 sm:grid-cols-2">
            {[
              { name: "supports_tools", label: "Supports tools" },
              { name: "supports_json_output", label: "Supports JSON output" },
              { name: "supports_reasoning", label: "Supports reasoning" },
              { name: "supports_parallel_tool", label: "Supports parallel tool calls" },
            ].map(({ name, label }) => (
              <label key={name} className="flex items-center gap-3 rounded-md border border-border bg-muted/30 px-4 py-3 text-sm font-medium transition-colors hover:bg-muted/50">
                <input type="hidden" name={name} value="off" />
                <input
                  type="checkbox"
                  name={name}
                  defaultChecked={name !== "supports_reasoning" && name !== "supports_parallel_tool"}
                  className="h-4 w-4 rounded border-input"
                />
                {label}
              </label>
            ))}
          </div>
        </div>
      </div>

      <div className="flex justify-end border-t border-border pt-6">
        <button
          type="submit"
          className="inline-flex items-center rounded-md bg-primary px-8 py-3 text-sm font-semibold text-primary-foreground shadow-sm transition-colors hover:bg-primary/90"
        >
          Save Provider
        </button>
      </div>
    </form>
  );
}

