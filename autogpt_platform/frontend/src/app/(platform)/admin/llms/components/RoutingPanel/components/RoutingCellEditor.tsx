"use client";

import type { LlmModelAdminResponse } from "@/app/api/__generated__/models/llmModelAdminResponse";
import { Select } from "@/components/atoms/Select/Select";
import { useLlmRegistryMutations } from "../../useLlmRegistryMutations";

interface Props {
  mode: string;
  tier: string;
  currentSlug: string | null;
  models: LlmModelAdminResponse[];
}

const FALL_THROUGH = "__fall_through__";

export function RoutingCellEditor({ mode, tier, currentSlug, models }: Props) {
  const { setRoute } = useLlmRegistryMutations();

  const options = [
    { value: FALL_THROUGH, label: "— falls through —" },
    ...models
      .filter((m) => m.is_enabled)
      .map((m) => ({
        value: m.slug,
        label: m.visibility === "HIDDEN" ? `${m.slug} (hidden)` : m.slug,
      })),
  ];

  function handleChange(value: string) {
    setRoute.mutate({
      data: {
        surface: "copilot",
        mode,
        tier,
        model_slug: value === FALL_THROUGH ? null : value,
      },
    });
  }

  return (
    <Select
      id={`route-${mode}-${tier}`}
      label={`${mode} ${tier} model`}
      hideLabel
      size="small"
      disabled={setRoute.isPending}
      value={currentSlug ?? FALL_THROUGH}
      onValueChange={handleChange}
      options={options}
    />
  );
}
