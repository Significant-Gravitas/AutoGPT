"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import type { LlmModel } from "@/app/api/__generated__/models/llmModel";
import { Button } from "@/components/atoms/Button/Button";
import { setRecommendedModelAction } from "../actions";
import { Star } from "@phosphor-icons/react";

export function RecommendedModelSelector({ models }: { models: LlmModel[] }) {
  const router = useRouter();
  const enabledModels = models.filter((m) => m.is_enabled);
  const currentRecommended = models.find((m) => m.is_recommended);

  const [selectedModelId, setSelectedModelId] = useState<string>(
    currentRecommended?.id || "",
  );
  const [isSaving, setIsSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const hasChanges = selectedModelId !== (currentRecommended?.id || "");

  async function handleSave() {
    if (!selectedModelId) return;

    setIsSaving(true);
    setError(null);
    try {
      const formData = new FormData();
      formData.set("model_id", selectedModelId);
      await setRecommendedModelAction(formData);
      router.refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to save");
    } finally {
      setIsSaving(false);
    }
  }

  return (
    <div className="rounded-lg border border-border bg-card p-4">
      <div className="mb-3 flex items-center gap-2">
        <Star size={20} weight="fill" className="text-amber-500" />
        <h3 className="text-sm font-semibold">Recommended Model</h3>
      </div>
      <p className="mb-3 text-xs text-muted-foreground">
        The recommended model is shown as the default suggestion in model
        selection dropdowns throughout the platform.
      </p>

      <div className="flex items-center gap-3">
        <select
          value={selectedModelId}
          onChange={(e) => setSelectedModelId(e.target.value)}
          className="flex-1 rounded-md border border-input bg-background px-3 py-2 text-sm"
          disabled={isSaving}
        >
          <option value="">-- Select a model --</option>
          {enabledModels.map((model) => (
            <option key={model.id} value={model.id}>
              {model.display_name} ({model.slug})
            </option>
          ))}
        </select>

        <Button
          type="button"
          variant="primary"
          size="small"
          onClick={handleSave}
          disabled={!hasChanges || !selectedModelId || isSaving}
        >
          {isSaving ? "Saving..." : "Save"}
        </Button>
      </div>

      {error && <p className="mt-2 text-xs text-destructive">{error}</p>}

      {currentRecommended && !hasChanges && (
        <p className="mt-2 text-xs text-muted-foreground">
          Currently set to:{" "}
          <span className="font-medium">{currentRecommended.display_name}</span>
        </p>
      )}
    </div>
  );
}
