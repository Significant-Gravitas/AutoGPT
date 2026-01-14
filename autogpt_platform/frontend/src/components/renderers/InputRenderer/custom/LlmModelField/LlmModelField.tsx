"use client";

import { FieldProps, RJSFSchema, getTemplate, getUiOptions } from "@rjsf/utils";
import { useMemo } from "react";
import { LlmModelPicker } from "./components/LlmModelPicker";
import { LlmModelMetadataMap } from "./types";
import { cleanUpHandleId, getHandleId, updateUiOption } from "../../helpers";
import { useEdgeStore } from "@/app/(platform)/build/stores/edgeStore";

type LlmModelSchema = RJSFSchema & {
  llm_model_metadata?: LlmModelMetadataMap;
};

export const LlmModelField = (props: FieldProps) => {
  const {
    schema,
    formData,
    onChange,
    disabled,
    readonly,
    fieldPathId,
    registry,
    uiSchema,
    required,
    name,
  } = props;

  const metadata = useMemo(() => {
    return (schema as LlmModelSchema)?.llm_model_metadata ?? {};
  }, [schema]);

  const models = useMemo(() => {
    return Object.values(metadata);
  }, [metadata]);

  const { isInputConnected } = useEdgeStore();

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

  const uiOptions = getUiOptions(uiSchema);
  const TitleFieldTemplate = getTemplate(
    "TitleFieldTemplate",
    registry,
    uiOptions,
  );
  const DescriptionFieldTemplate = getTemplate(
    "DescriptionFieldTemplate",
    registry,
    uiOptions,
  );

  const fieldId = fieldPathId?.$id ?? props.id ?? "llm-model-field";
  const handleId = getHandleId({
    uiOptions,
    id: fieldId,
    schema: schema,
  });
  const updatedUiSchema = updateUiOption(uiSchema, {
    handleId: handleId,
  });
  const title = schema.title || name || "LLM Model";

  const { nodeId } = registry.formContext ?? {};

  const isHandleConnected = nodeId
    ? isInputConnected(nodeId, cleanUpHandleId(handleId))
    : false;

  return (
    <div className="-my-3 flex flex-col gap-1">
      <div className="flex items-center gap-2">
        <TitleFieldTemplate
          id={fieldId}
          title={title}
          required={required}
          schema={schema}
          uiSchema={updatedUiSchema}
          registry={registry}
        />
        <DescriptionFieldTemplate
          id={fieldId}
          description={schema.description || ""}
          schema={schema}
          registry={registry}
        />
      </div>
      {!isHandleConnected && (
        <LlmModelPicker
          models={models}
          selectedModel={selectedModel}
          recommendedModel={recommendedModel}
          onSelect={(value) => onChange(value, fieldPathId?.path)}
          disabled={disabled || readonly}
        />
      )}
    </div>
  );
};
