"use client";

import type { CreateLlmModelRequest } from "@/app/api/__generated__/models/createLlmModelRequest";
import type { LlmModelAdminResponse } from "@/app/api/__generated__/models/llmModelAdminResponse";
import { useState } from "react";
import { useLlmRegistryMutations } from "../useLlmRegistryMutations";

export interface ModelFormValues {
  slug: string;
  display_name: string;
  description: string;
  provider_name: string;
  creator_id: string;
  context_window: string;
  max_output_tokens: string;
  price_tier: string;
  visibility: string;
  min_subscription_tier: string;
  fallback_model_slug: string;
  is_enabled: boolean;
  is_recommended: boolean;
  supports_tools: boolean;
  supports_json_output: boolean;
  supports_reasoning: boolean;
  supports_parallel_tool_calls: boolean;
}

const NONE = "__none__";

export function emptyFormValues(): ModelFormValues {
  return {
    slug: "",
    display_name: "",
    description: "",
    provider_name: "",
    creator_id: NONE,
    context_window: "128000",
    max_output_tokens: "",
    price_tier: "1",
    visibility: "GA",
    min_subscription_tier: NONE,
    fallback_model_slug: NONE,
    is_enabled: true,
    is_recommended: false,
    supports_tools: false,
    supports_json_output: false,
    supports_reasoning: false,
    supports_parallel_tool_calls: false,
  };
}

export function valuesFromModel(model: LlmModelAdminResponse): ModelFormValues {
  return {
    slug: model.slug,
    display_name: model.display_name,
    description: model.description ?? "",
    provider_name: "",
    creator_id: model.creator?.id ?? NONE,
    context_window: String(model.context_window),
    max_output_tokens: model.max_output_tokens
      ? String(model.max_output_tokens)
      : "",
    price_tier: String(model.price_tier),
    visibility: model.visibility,
    min_subscription_tier: model.min_subscription_tier ?? NONE,
    fallback_model_slug: model.fallback_model_slug ?? NONE,
    is_enabled: model.is_enabled,
    is_recommended: model.is_recommended,
    supports_tools: model.supports_tools,
    supports_json_output: model.supports_json_output,
    supports_reasoning: model.supports_reasoning,
    supports_parallel_tool_calls: model.supports_parallel_tool_calls,
  };
}

function optional(value: string): string | null {
  return value === NONE || value === "" ? null : value;
}

export function useModelFormDialog(args: {
  editing: LlmModelAdminResponse | null;
  onClose: () => void;
}) {
  const { editing, onClose } = args;
  const [values, setValues] = useState<ModelFormValues>(() =>
    editing ? valuesFromModel(editing) : emptyFormValues(),
  );
  const [validationError, setValidationError] = useState("");
  const { createModel, updateModel } = useLlmRegistryMutations();

  function setField<K extends keyof ModelFormValues>(
    key: K,
    value: ModelFormValues[K],
  ) {
    setValues((prev) => ({ ...prev, [key]: value }));
  }

  function buildPayload() {
    const contextWindow = Number(values.context_window);
    if (!Number.isFinite(contextWindow) || contextWindow <= 0) {
      return { error: "Context window must be a positive number" };
    }
    const maxOutput = values.max_output_tokens
      ? Number(values.max_output_tokens)
      : null;
    if (maxOutput !== null && (!Number.isFinite(maxOutput) || maxOutput <= 0)) {
      return { error: "Max output tokens must be a positive number" };
    }
    const shared = {
      display_name: values.display_name,
      description: values.description || null,
      creator_id: optional(values.creator_id),
      context_window: contextWindow,
      max_output_tokens: maxOutput,
      price_tier: Number(values.price_tier),
      is_enabled: values.is_enabled,
      is_recommended: values.is_recommended,
      visibility: values.visibility as CreateLlmModelRequest["visibility"],
      min_subscription_tier: optional(
        values.min_subscription_tier,
      ) as CreateLlmModelRequest["min_subscription_tier"],
      fallback_model_slug: optional(values.fallback_model_slug),
      supports_tools: values.supports_tools,
      supports_json_output: values.supports_json_output,
      supports_reasoning: values.supports_reasoning,
      supports_parallel_tool_calls: values.supports_parallel_tool_calls,
    };
    return { shared };
  }

  async function handleSubmit() {
    const built = buildPayload();
    if ("error" in built) {
      setValidationError(built.error ?? "Invalid input");
      return;
    }
    setValidationError("");
    if (editing) {
      await updateModel.mutateAsync({
        slug: editing.slug,
        data: built.shared,
      });
    } else {
      if (!values.slug || !values.provider_name) {
        setValidationError("Slug and provider are required");
        return;
      }
      await createModel.mutateAsync({
        data: {
          ...built.shared,
          slug: values.slug,
          provider_name: values.provider_name,
        },
      });
    }
    onClose();
  }

  const isPending = createModel.isPending || updateModel.isPending;

  return { values, setField, handleSubmit, isPending, validationError, NONE };
}
