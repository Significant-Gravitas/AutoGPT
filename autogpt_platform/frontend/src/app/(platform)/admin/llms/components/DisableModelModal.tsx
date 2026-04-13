"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { Button } from "@/components/atoms/Button/Button";
import type { LlmModel } from "../types";
import { toggleLlmModelAction, fetchLlmModelUsage } from "../actions";

export function DisableModelModal({
  model,
  availableModels,
}: {
  model: LlmModel;
  availableModels: LlmModel[];
}) {
  const router = useRouter();
  const [open, setOpen] = useState(false);
  const [isDisabling, setIsDisabling] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [usageCount, setUsageCount] = useState<number | null>(null);
  const [selectedMigration, setSelectedMigration] = useState<string>("");
  const [migrationReason, setMigrationReason] = useState("");
  const [customCreditCost, setCustomCreditCost] = useState<string>("");

  const migrationOptions = availableModels.filter(
    (m) => m.id !== model.id && m.is_enabled,
  );

  async function fetchUsage() {
    try {
      const usage = await fetchLlmModelUsage(model.slug);
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
      router.refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to disable model");
    } finally {
      setIsDisabling(false);
    }
  }

  function resetState() {
    setError(null);
    setSelectedMigration("");
    setMigrationReason("");
    setCustomCreditCost("");
    setUsageCount(null);
  }

  const hasUsage = usageCount !== null && usageCount > 0;
  const isLoading = usageCount === null && !model.is_recommended;

  return (
    <Dialog
      title="Disable Model"
      controlled={{
        isOpen: open,
        set: async (isOpen) => {
          setOpen(isOpen);
          if (isOpen) {
            resetState();
            if (!model.is_recommended) {
              await fetchUsage();
            }
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
        {model.is_recommended ? (
          <RecommendedModelBlock model={model} onClose={() => setOpen(false)} />
        ) : (
          <DisableForm
            model={model}
            usageCount={usageCount}
            isLoading={isLoading}
            hasUsage={hasUsage}
            migrationOptions={migrationOptions}
            selectedMigration={selectedMigration}
            setSelectedMigration={setSelectedMigration}
            migrationReason={migrationReason}
            setMigrationReason={setMigrationReason}
            customCreditCost={customCreditCost}
            setCustomCreditCost={setCustomCreditCost}
            isDisabling={isDisabling}
            error={error}
            onClose={() => {
              setOpen(false);
              resetState();
            }}
            onSubmit={handleDisable}
          />
        )}
      </Dialog.Content>
    </Dialog>
  );
}

function RecommendedModelBlock({
  model,
  onClose,
}: {
  model: LlmModel;
  onClose: () => void;
}) {
  return (
    <div className="space-y-4">
      <div className="rounded-lg border border-destructive/30 bg-destructive/10 p-4">
        <div className="flex items-start gap-3">
          <div className="flex-shrink-0 text-destructive">🔒</div>
          <div className="text-sm">
            <p className="font-semibold text-destructive">
              Cannot disable the recommended model
            </p>
            <p className="mt-1 text-foreground">
              <span className="font-medium">{model.display_name}</span>{" "}
              <span className="text-muted-foreground">({model.slug})</span> is
              currently set as the recommended model.
            </p>
            <p className="mt-2 text-muted-foreground">
              Change the recommended model to a different enabled model before
              disabling this one.
            </p>
          </div>
        </div>
      </div>
      <Dialog.Footer>
        <Button variant="ghost" size="small" onClick={onClose}>
          Close
        </Button>
      </Dialog.Footer>
    </div>
  );
}

interface DisableFormProps {
  model: LlmModel;
  usageCount: number | null;
  isLoading: boolean;
  hasUsage: boolean;
  migrationOptions: LlmModel[];
  selectedMigration: string;
  setSelectedMigration: (v: string) => void;
  migrationReason: string;
  setMigrationReason: (v: string) => void;
  customCreditCost: string;
  setCustomCreditCost: (v: string) => void;
  isDisabling: boolean;
  error: string | null;
  onClose: () => void;
  onSubmit: (formData: FormData) => Promise<void>;
}

function DisableForm({
  model,
  usageCount,
  isLoading,
  hasUsage,
  migrationOptions,
  selectedMigration,
  setSelectedMigration,
  migrationReason,
  setMigrationReason,
  customCreditCost,
  setCustomCreditCost,
  isDisabling,
  error,
  onClose,
  onSubmit,
}: DisableFormProps) {
  const submitDisabled =
    isDisabling ||
    isLoading ||
    (hasUsage && !selectedMigration) ||
    (hasUsage && migrationOptions.length === 0);

  return (
    <div className="space-y-4">
      <div className="text-sm text-muted-foreground">
        Disabling a model will hide it from users when creating new workflows.
      </div>

      <div className="rounded-lg border border-amber-500/30 bg-amber-500/10 p-4">
        <div className="flex items-start gap-3">
          <div className="flex-shrink-0 text-amber-600">⚠️</div>
          <div className="text-sm text-foreground">
            <p className="font-semibold">You are about to disable:</p>
            <p className="mt-1">
              <span className="font-medium">{model.display_name}</span>{" "}
              <span className="text-muted-foreground">({model.slug})</span>
            </p>
            {isLoading ? (
              <p className="mt-2 text-muted-foreground">
                Loading usage data...
              </p>
            ) : hasUsage ? (
              <p className="mt-2 font-semibold text-amber-700">
                Impact: {usageCount} block{usageCount !== 1 ? "s" : ""}{" "}
                currently use this model — migration required
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
          <div className="text-sm">
            <p className="font-medium">Migration required</p>
            <p className="mt-1 text-muted-foreground">
              Workflows using this model must be migrated to a replacement
              before disabling. This creates a revertible migration record.
            </p>
          </div>

          <div className="space-y-4 border-t border-border pt-4">
            <label className="block text-sm font-medium">
              <span className="mb-2 block">
                Replacement Model <span className="text-destructive">*</span>
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
                  No other enabled models available. Enable another model before
                  disabling this one.
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
                Override pricing for migrated workflows.
              </p>
            </label>
          </div>
        </div>
      )}

      <form action={onSubmit} className="space-y-4">
        <input type="hidden" name="model_id" value={model.slug} />
        <input type="hidden" name="is_enabled" value="false" />
        {hasUsage && selectedMigration && (
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
            onClick={onClose}
            disabled={isDisabling}
          >
            Cancel
          </Button>
          <Button
            type="submit"
            variant="primary"
            size="small"
            disabled={submitDisabled}
          >
            {isDisabling
              ? "Disabling..."
              : hasUsage && selectedMigration
                ? "Disable & Migrate"
                : "Disable Model"}
          </Button>
        </Dialog.Footer>
      </form>
    </div>
  );
}
