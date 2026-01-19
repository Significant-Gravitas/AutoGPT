"use client";

import { useState } from "react";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { Button } from "@/components/atoms/Button/Button";
import type { LlmModel } from "@/app/api/__generated__/models/llmModel";
import { toggleLlmModelAction, fetchLlmModelUsage } from "../actions";

export function DisableModelModal({
  model,
  availableModels,
}: {
  model: LlmModel;
  availableModels: LlmModel[];
}) {
  const [open, setOpen] = useState(false);
  const [isDisabling, setIsDisabling] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [usageCount, setUsageCount] = useState<number | null>(null);
  const [selectedMigration, setSelectedMigration] = useState<string>("");
  const [wantsMigration, setWantsMigration] = useState(false);
  const [migrationReason, setMigrationReason] = useState("");
  const [customCreditCost, setCustomCreditCost] = useState<string>("");

  // Filter out the current model and disabled models from replacement options
  const migrationOptions = availableModels.filter(
    (m) => m.id !== model.id && m.is_enabled,
  );

  async function fetchUsage() {
    try {
      const usage = await fetchLlmModelUsage(model.id);
      setUsageCount(usage.node_count);
    } catch {
      setUsageCount(null);
    }
  }

  async function handleDisable(formData: FormData) {
    setIsDisabling(true);
    setError(null);
    try {
      await toggleLlmModelAction(formData);
      setOpen(false);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to disable model");
    } finally {
      setIsDisabling(false);
    }
  }

  function resetState() {
    setError(null);
    setSelectedMigration("");
    setWantsMigration(false);
    setMigrationReason("");
    setCustomCreditCost("");
  }

  const hasUsage = usageCount !== null && usageCount > 0;

  return (
    <Dialog
      title="Disable Model"
      controlled={{
        isOpen: open,
        set: async (isOpen) => {
          setOpen(isOpen);
          if (isOpen) {
            setUsageCount(null);
            resetState();
            await fetchUsage();
          }
        },
      }}
      styling={{ maxWidth: "600px" }}
    >
      <Dialog.Trigger>
        <Button
          type="button"
          variant="outline"
          size="small"
          className="min-w-0"
        >
          Disable
        </Button>
      </Dialog.Trigger>
      <Dialog.Content>
        <div className="mb-4 text-sm text-muted-foreground">
          Disabling a model will hide it from users when creating new workflows.
        </div>

        <div className="space-y-4">
          <div className="rounded-lg border border-amber-500/30 bg-amber-500/10 p-4 dark:border-amber-400/30 dark:bg-amber-400/10">
            <div className="flex items-start gap-3">
              <div className="flex-shrink-0 text-amber-600 dark:text-amber-400">
                ⚠️
              </div>
              <div className="text-sm text-foreground">
                <p className="font-semibold">You are about to disable:</p>
                <p className="mt-1">
                  <span className="font-medium">{model.display_name}</span>{" "}
                  <span className="text-muted-foreground">({model.slug})</span>
                </p>
                {usageCount === null ? (
                  <p className="mt-2 text-muted-foreground">
                    Loading usage data...
                  </p>
                ) : usageCount > 0 ? (
                  <p className="mt-2 font-semibold">
                    Impact: {usageCount} block{usageCount !== 1 ? "s" : ""}{" "}
                    currently use this model
                  </p>
                ) : (
                  <p className="mt-2 text-muted-foreground">
                    No workflows are currently using this model.
                  </p>
                )}
              </div>
            </div>
          </div>

          {hasUsage && (
            <div className="space-y-4 rounded-lg border border-border bg-muted/50 p-4">
              <label className="flex items-start gap-3">
                <input
                  type="checkbox"
                  checked={wantsMigration}
                  onChange={(e) => {
                    setWantsMigration(e.target.checked);
                    if (!e.target.checked) {
                      setSelectedMigration("");
                    }
                  }}
                  className="mt-1"
                />
                <div className="text-sm">
                  <span className="font-medium">
                    Migrate existing workflows to another model
                  </span>
                  <p className="mt-1 text-muted-foreground">
                    Creates a revertible migration record. If unchecked,
                    existing workflows will use automatic fallback to an enabled
                    model from the same provider.
                  </p>
                </div>
              </label>

              {wantsMigration && (
                <div className="space-y-4 border-t border-border pt-4">
                  <label className="block text-sm font-medium">
                    <span className="mb-2 block">
                      Replacement Model{" "}
                      <span className="text-destructive">*</span>
                    </span>
                    <select
                      required
                      value={selectedMigration}
                      onChange={(e) => setSelectedMigration(e.target.value)}
                      className="w-full rounded border border-input bg-background p-2 text-sm"
                    >
                      <option value="">-- Choose a replacement model --</option>
                      {migrationOptions.map((m) => (
                        <option key={m.id} value={m.slug}>
                          {m.display_name} ({m.slug})
                        </option>
                      ))}
                    </select>
                    {migrationOptions.length === 0 && (
                      <p className="mt-2 text-xs text-destructive">
                        No other enabled models available for migration.
                      </p>
                    )}
                  </label>

                  <label className="block text-sm font-medium">
                    <span className="mb-2 block">
                      Migration Reason{" "}
                      <span className="font-normal text-muted-foreground">
                        (optional)
                      </span>
                    </span>
                    <input
                      type="text"
                      value={migrationReason}
                      onChange={(e) => setMigrationReason(e.target.value)}
                      placeholder="e.g., Provider outage, Cost reduction"
                      className="w-full rounded border border-input bg-background p-2 text-sm"
                    />
                    <p className="mt-1 text-xs text-muted-foreground">
                      Helps track why the migration was made
                    </p>
                  </label>

                  <label className="block text-sm font-medium">
                    <span className="mb-2 block">
                      Custom Credit Cost{" "}
                      <span className="font-normal text-muted-foreground">
                        (optional)
                      </span>
                    </span>
                    <input
                      type="number"
                      min="0"
                      value={customCreditCost}
                      onChange={(e) => setCustomCreditCost(e.target.value)}
                      placeholder="Leave blank to use target model's cost"
                      className="w-full rounded border border-input bg-background p-2 text-sm"
                    />
                    <p className="mt-1 text-xs text-muted-foreground">
                      Override pricing for migrated workflows. When set, billing
                      will use this cost instead of the target model&apos;s
                      cost.
                    </p>
                  </label>
                </div>
              )}
            </div>
          )}

          <form action={handleDisable} className="space-y-4">
            <input type="hidden" name="model_id" value={model.id} />
            <input type="hidden" name="is_enabled" value="false" />
            {wantsMigration && selectedMigration && (
              <>
                <input
                  type="hidden"
                  name="migrate_to_slug"
                  value={selectedMigration}
                />
                {migrationReason && (
                  <input
                    type="hidden"
                    name="migration_reason"
                    value={migrationReason}
                  />
                )}
                {customCreditCost && (
                  <input
                    type="hidden"
                    name="custom_credit_cost"
                    value={customCreditCost}
                  />
                )}
              </>
            )}

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
                  resetState();
                }}
                disabled={isDisabling}
              >
                Cancel
              </Button>
              <Button
                type="submit"
                variant="primary"
                size="small"
                disabled={
                  isDisabling ||
                  (wantsMigration && !selectedMigration) ||
                  usageCount === null
                }
              >
                {isDisabling
                  ? "Disabling..."
                  : wantsMigration && selectedMigration
                    ? "Disable & Migrate"
                    : "Disable Model"}
              </Button>
            </Dialog.Footer>
          </form>
        </div>
      </Dialog.Content>
    </Dialog>
  );
}
