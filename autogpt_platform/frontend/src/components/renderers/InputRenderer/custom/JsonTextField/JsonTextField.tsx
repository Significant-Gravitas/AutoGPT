"use client";

import {
  getFieldDomId,
  useFieldAccessibilityContext,
} from "../../field-accessibility";
import { FieldError } from "../../base/standard/FieldError";

import { FieldProps, getTemplate, getUiOptions, titleId } from "@rjsf/utils";
import { Input } from "@/components/atoms/Input/Input";
import { Button } from "@/components/atoms/Button/Button";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/atoms/Tooltip/BaseTooltip";
import { InputExpanderModal } from "../../base/standard/widgets/TextInput/TextInputExpanderModal";
import { getHandleId, updateUiOption, cleanUpHandleId } from "../../helpers";
import { useJsonTextField } from "./useJsonTextField";
import { getPlaceholder } from "./helpers";
import { ArrowExpandIcon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";

export const JsonTextField = (props: FieldProps) => {
  const {
    formData,
    onChange,
    schema,
    registry,
    uiSchema,
    required,
    name,
    fieldPathId,
  } = props;

  const uiOptions = getUiOptions(uiSchema);

  const TitleFieldTemplate = getTemplate(
    "TitleFieldTemplate",
    registry,
    uiOptions,
  );

  const fieldId = fieldPathId?.$id ?? props.id ?? "json-field";

  const handleId = getHandleId({
    uiOptions,
    id: fieldId,
    schema: schema,
  });

  const updatedUiSchema = updateUiOption(uiSchema, {
    handleId: handleId,
  });

  const {
    textValue,
    isModalOpen,
    handleChange,
    handleModalOpen,
    handleModalClose,
    handleModalSave,
  } = useJsonTextField({
    formData,
    onChange,
    path: fieldPathId?.path,
  });

  const placeholder = getPlaceholder(schema);
  const title = schema.title || name || "JSON Value";
  const domId = getFieldDomId(fieldId, registry.formContext);
  const fieldAccessibility = useFieldAccessibilityContext(fieldId);
  const labelId = getFieldDomId(titleId(fieldId), registry.formContext);
  const errorId = fieldAccessibility?.errorId ?? `${domId}-error`;
  const descriptionId = schema.description
    ? `${domId}-json-description`
    : undefined;

  return (
    <div className="flex flex-col gap-2">
      <TitleFieldTemplate
        id={titleId(fieldId)}
        title={title}
        required={required}
        schema={schema}
        uiSchema={updatedUiSchema}
        registry={registry}
      />
      <div className="nodrag relative flex items-center gap-2">
        <Input
          id={domId}
          aria-labelledby={labelId}
          aria-describedby={[
            descriptionId ?? fieldAccessibility?.descriptionId,
            errorId,
          ]
            .filter(Boolean)
            .join(" ")}
          hideLabel={true}
          type="textarea"
          label={title}
          size="small"
          wrapperClassName="mb-0 flex-1 "
          value={textValue}
          onChange={handleChange}
          placeholder={placeholder}
          required={required}
          disabled={props.disabled}
          className="min-h-[60px] pr-8 font-mono text-xs"
        />

        <Tooltip delayDuration={0}>
          <TooltipTrigger asChild>
            <Button
              variant="ghost"
              size="icon"
              onClick={handleModalOpen}
              type="button"
              className="p-1"
              aria-label="Expand JSON input"
            >
              <Icon icon={ArrowExpandIcon} className="size-4" />
            </Button>
          </TooltipTrigger>
          <TooltipContent>Expand input</TooltipContent>
        </Tooltip>
      </div>
      {schema.description && (
        <span id={descriptionId} className="text-xs text-gray-500">
          {schema.description}
        </span>
      )}

      {!fieldAccessibility && (
        <FieldError
          nodeId={registry.formContext.nodeId}
          fieldId={cleanUpHandleId(fieldId)}
          id={errorId}
        />
      )}
      <InputExpanderModal
        isOpen={isModalOpen}
        onClose={handleModalClose}
        onSave={handleModalSave}
        title={`Edit ${title}`}
        description={schema.description || "Enter valid JSON"}
        defaultValue={textValue}
        placeholder={placeholder}
        inputType="json"
      />
    </div>
  );
};

export default JsonTextField;
