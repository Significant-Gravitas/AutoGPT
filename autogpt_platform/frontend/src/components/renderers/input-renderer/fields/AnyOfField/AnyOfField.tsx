import React from "react";
import { FieldProps, RJSFSchema } from "@rjsf/utils";

import { Text } from "@/components/atoms/Text/Text";
import { Switch } from "@/components/atoms/Switch/Switch";
import { Select } from "@/components/atoms/Select/Select";
import {
  InputType,
  mapJsonSchemaTypeToInputType,
} from "@/app/(platform)/build/components/FlowEditor/nodes/helpers";

import { InfoIcon } from "@phosphor-icons/react";
import { useAnyOfField } from "./useAnyOfField";
import NodeHandle from "@/app/(platform)/build/components/FlowEditor/handlers/NodeHandle";
import { useEdgeStore } from "@/app/(platform)/build/stores/edgeStore";
import { generateHandleId } from "@/app/(platform)/build/components/FlowEditor/handlers/helpers";
import { getTypeDisplayInfo } from "@/app/(platform)/build/components/FlowEditor/nodes/helpers";
import merge from "lodash/merge";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/atoms/Tooltip/BaseTooltip";
import { cn } from "@/lib/utils";

type TypeOption = {
  type: string;
  title: string;
  index: number;
  format?: string;
  enum?: any[];
  secret?: boolean;
  schema: RJSFSchema;
};

export const AnyOfField = ({
  schema,
  formData,
  onChange,
  name,
  idSchema,
  formContext,
  registry,
  uiSchema,
  disabled,
  onBlur,
  onFocus,
}: FieldProps) => {
  const handleId = generateHandleId(idSchema.$id ?? "");
  const updatedFormContexrt = { ...formContext, fromAnyOf: true };

  const { nodeId, showHandles = true } = updatedFormContexrt;
  const { isInputConnected } = useEdgeStore();
  const isConnected = showHandles ? isInputConnected(nodeId, handleId) : false;
  const {
    isNullableType,
    nonNull,
    selectedType,
    handleTypeChange,
    handleNullableToggle,
    handleValueChange,
    currentTypeOption,
    isEnabled,
    typeOptions,
  } = useAnyOfField(schema, formData, onChange);

  const renderInput = (typeOption: TypeOption) => {
    const optionSchema = (typeOption.schema || {
      type: typeOption.type,
      format: typeOption.format,
      secret: typeOption.secret,
      enum: typeOption.enum,
    }) as RJSFSchema;
    const inputType = mapJsonSchemaTypeToInputType(optionSchema);

    // Help us to tell the field under the anyOf field that you are a part of anyOf field.
    // We can't use formContext in this case that's why we are using this.
    // We could use context api here, but i think it's better to keep it simple.
    const uiSchemaFromAnyOf = merge({}, uiSchema, {
      "ui:options": { fromAnyOf: true },
    });

    // We are using SchemaField to render the field recursively.
    if (inputType === InputType.ARRAY_EDITOR) {
      const SchemaField = registry.fields.SchemaField;
      return (
        <div className="-ml-2">
          <SchemaField
            schema={optionSchema}
            formData={formData}
            idSchema={idSchema}
            uiSchema={uiSchemaFromAnyOf}
            onChange={handleValueChange}
            onBlur={onBlur}
            onFocus={onFocus}
            name={name}
            registry={registry}
            disabled={disabled}
            formContext={updatedFormContexrt}
          />
        </div>
      );
    }

    const SchemaField = registry.fields.SchemaField;
    return (
      <div className="-ml-2">
        <SchemaField
          schema={optionSchema}
          formData={formData}
          idSchema={idSchema}
          uiSchema={uiSchemaFromAnyOf}
          onChange={handleValueChange}
          onBlur={onBlur}
          onFocus={onFocus}
          name={name}
          registry={registry}
          disabled={disabled}
          formContext={updatedFormContexrt}
        />
      </div>
    );
  };

  // I am doing this, because we need different UI for optional types.
  if (isNullableType && nonNull) {
    const { displayType, colorClass } = getTypeDisplayInfo(nonNull);

    return (
      <div className="mb-0 flex flex-col">
        <div className="flex items-center justify-between gap-2">
          <div
            className={cn("flex items-center gap-1", showHandles && "-ml-2")}
          >
            {showHandles && (
              <NodeHandle
                handleId={handleId}
                isConnected={isConnected}
                side="left"
              />
            )}
            <Text
              variant={formContext.size === "small" ? "body" : "body-medium"}
            >
              {name.charAt(0).toUpperCase() + name.slice(1)}
            </Text>
            <Text variant="small" className={colorClass}>
              ({displayType} | null)
            </Text>
          </div>
          {!isConnected && (
            <Switch
              className="z-10"
              checked={isEnabled}
              onCheckedChange={handleNullableToggle}
            />
          )}
        </div>
        <div>{!isConnected && isEnabled && renderInput(nonNull)}</div>
      </div>
    );
  }

  return (
    <div className="mb-0 flex flex-col">
      <div className={cn("flex items-center gap-1", showHandles && "-ml-2")}>
        {showHandles && (
          <NodeHandle
            handleId={handleId}
            isConnected={isConnected}
            side="left"
          />
        )}
        <Text variant={formContext.size === "small" ? "body" : "body-medium"}>
          {name.charAt(0).toUpperCase() + name.slice(1)}
        </Text>
        {!isConnected && (
          <Select
            label=""
            id={`${name}-type-select`}
            hideLabel={true}
            value={selectedType}
            onValueChange={handleTypeChange}
            options={typeOptions.map((o) => {
              const { displayType } = getTypeDisplayInfo(o);
              return { value: o.type, label: displayType };
            })}
            size="small"
            wrapperClassName="!mb-0 "
            className="h-6 w-fit gap-1 pl-3 pr-2"
          />
        )}

        {schema.description && (
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <span
                  style={{ marginLeft: 6, cursor: "pointer" }}
                  aria-label="info"
                  tabIndex={0}
                >
                  <InfoIcon />
                </span>
              </TooltipTrigger>
              <TooltipContent>{schema.description}</TooltipContent>
            </Tooltip>
          </TooltipProvider>
        )}
      </div>

      {!isConnected && currentTypeOption && renderInput(currentTypeOption)}
    </div>
  );
};
