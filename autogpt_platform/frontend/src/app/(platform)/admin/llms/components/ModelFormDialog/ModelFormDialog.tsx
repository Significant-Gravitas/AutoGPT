"use client";

import type { LlmCreatorAdminResponse } from "@/app/api/__generated__/models/llmCreatorAdminResponse";
import type { LlmModelAdminResponse } from "@/app/api/__generated__/models/llmModelAdminResponse";
import type { LlmProviderAdminResponse } from "@/app/api/__generated__/models/llmProviderAdminResponse";
import { Button } from "@/components/atoms/Button/Button";
import { Input } from "@/components/atoms/Input/Input";
import { Select } from "@/components/atoms/Select/Select";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { CapabilityToggles } from "./components/CapabilityToggles";
import { useModelFormDialog } from "./useModelFormDialog";

interface Props {
  open: boolean;
  editing: LlmModelAdminResponse | null;
  providers: LlmProviderAdminResponse[];
  creators: LlmCreatorAdminResponse[];
  models: LlmModelAdminResponse[];
  onClose: () => void;
}

const VISIBILITIES = ["GA", "EMPLOYEES", "ADMINS", "HIDDEN"];
const TIERS = ["NO_TIER", "BASIC", "PRO", "MAX", "BUSINESS", "ENTERPRISE"];

export function ModelFormDialog({
  open,
  editing,
  providers,
  creators,
  models,
  onClose,
}: Props) {
  const { values, setField, handleSubmit, isPending, validationError, NONE } =
    useModelFormDialog({ editing, onClose });

  const noneOption = { value: NONE, label: "— none —" };

  return (
    <Dialog
      title={editing ? `Edit ${editing.slug}` : "Add model"}
      styling={{ maxWidth: "40rem" }}
      controlled={{
        isOpen: open,
        set: (next) => (next ? undefined : onClose()),
      }}
    >
      <Dialog.Content>
        <div className="flex max-h-[70vh] flex-col gap-3 overflow-y-auto pr-1">
          {!editing && (
            <Input
              id="model-slug"
              label="Slug"
              placeholder="provider/model-name"
              value={values.slug}
              onChange={(e) => setField("slug", e.target.value)}
            />
          )}
          <Input
            id="model-display-name"
            label="Display name"
            value={values.display_name}
            onChange={(e) => setField("display_name", e.target.value)}
          />
          <Input
            id="model-description"
            label="Description"
            value={values.description}
            onChange={(e) => setField("description", e.target.value)}
          />
          {!editing && (
            <Select
              id="model-provider"
              label="Provider"
              placeholder="Select provider"
              value={values.provider_name}
              onValueChange={(v) => setField("provider_name", v)}
              options={providers.map((p) => ({
                value: p.name,
                label: p.display_name,
              }))}
            />
          )}
          <Select
            id="model-creator"
            label="Creator"
            value={values.creator_id}
            onValueChange={(v) => setField("creator_id", v)}
            options={[
              noneOption,
              ...creators.map((c) => ({ value: c.id, label: c.display_name })),
            ]}
          />
          <div className="grid grid-cols-2 gap-3">
            <Input
              id="model-context-window"
              label="Context window"
              type="number"
              value={values.context_window}
              onChange={(e) => setField("context_window", e.target.value)}
            />
            <Input
              id="model-max-output"
              label="Max output tokens"
              type="number"
              placeholder="defaults to context window"
              value={values.max_output_tokens}
              onChange={(e) => setField("max_output_tokens", e.target.value)}
            />
          </div>
          <div className="grid grid-cols-3 gap-3">
            <Select
              id="model-price-tier"
              label="Price tier"
              value={values.price_tier}
              onValueChange={(v) => setField("price_tier", v)}
              options={[
                { value: "1", label: "1 — cheap" },
                { value: "2", label: "2 — medium" },
                { value: "3", label: "3 — expensive" },
              ]}
            />
            <Select
              id="model-visibility"
              label="Visibility"
              value={values.visibility}
              onValueChange={(v) => setField("visibility", v)}
              options={VISIBILITIES.map((v) => ({ value: v, label: v }))}
            />
            <Select
              id="model-min-tier"
              label="Min subscription"
              value={values.min_subscription_tier}
              onValueChange={(v) => setField("min_subscription_tier", v)}
              options={[
                noneOption,
                ...TIERS.map((t) => ({ value: t, label: t })),
              ]}
            />
          </div>
          <Select
            id="model-fallback"
            label="Fallback model"
            value={values.fallback_model_slug}
            onValueChange={(v) => setField("fallback_model_slug", v)}
            options={[
              noneOption,
              ...models
                .filter((m) => m.slug !== editing?.slug)
                .map((m) => ({ value: m.slug, label: m.slug })),
            ]}
          />
          <CapabilityToggles values={values} setField={setField} />
          {validationError && (
            <p className="text-sm text-red-600">{validationError}</p>
          )}
        </div>
        <Dialog.Footer>
          <Button variant="secondary" onClick={onClose} disabled={isPending}>
            Cancel
          </Button>
          <Button onClick={handleSubmit} loading={isPending}>
            {editing ? "Save changes" : "Create model"}
          </Button>
        </Dialog.Footer>
      </Dialog.Content>
    </Dialog>
  );
}
