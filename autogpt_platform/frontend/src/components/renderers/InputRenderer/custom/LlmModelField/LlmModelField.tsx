"use client";

import { FieldProps, RJSFSchema } from "@rjsf/utils";
import { useMemo } from "react";
import { LlmModelPicker } from "./components/LlmModelPicker";
import { LlmModelMetadataMap } from "./types";

type LlmModelSchema = RJSFSchema & {
  llm_model_metadata?: LlmModelMetadataMap;
};

export const LlmModelField = (props: FieldProps) => {
  const { schema, formData, onChange, disabled, readonly, fieldPathId } = props;

  const metadata = useMemo(() => {
    return (schema as LlmModelSchema)?.llm_model_metadata ?? {};
  }, [schema]);

  const models = useMemo(() => {
    return Object.values(metadata);
  }, [metadata]);

  const selectedName =
    typeof formData === "string"
      ? formData
      : typeof schema.default === "string"
        ? schema.default
        : "";

  const selectedModel = selectedName
    ? (metadata[selectedName] ??
      models.find((model) => model.name === selectedName))
    : undefined;

  const recommendedName =
    typeof schema.default === "string" ? schema.default : models[0]?.name;

  const recommendedModel =
    recommendedName && metadata[recommendedName]
      ? metadata[recommendedName]
      : undefined;

  if (models.length === 0) {
    return null;
  }

  return (
    <LlmModelPicker
      models={models}
      selectedModel={selectedModel}
      recommendedModel={recommendedModel}
      onSelect={(value) => onChange(value, fieldPathId?.path)}
      disabled={disabled || readonly}
    />
  );
};
