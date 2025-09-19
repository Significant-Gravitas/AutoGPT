import React from "react";
import { FieldProps, RJSFSchema } from "@rjsf/utils";

import { Text } from "@/components/atoms/Text/Text";
import { Switch } from "@/components/atoms/Switch/Switch";
import { Select } from "@/components/atoms/Select/Select";
import { InputRenderer, InputType } from "../../InputRenderer";
import { mapJsonSchemaTypeToInputType, extractOptions } from "../../helpers";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { InfoIcon } from "@phosphor-icons/react";
import { useAnyOfField } from "./useAnyOfField";
import NodeHandle from "../../../handlers/NodeHandle";
import { useEdgeStore } from "../../../../store/edgeStore";
import { generateHandleId } from "../../../handlers/helpers";

type TypeOption = {
  type: string;
  title: string;
  index: number;
  format?: string;
  enum?: any[];
  secret?: boolean;
};

export const AnyOfField = ({
  schema,
  formData,
  onChange,
  name,
  idSchema,
  formContext,
}: FieldProps) => {
  const fieldKey = generateHandleId(idSchema.$id ?? "");

  const { nodeId } = formContext;
  const { isInputConnected } = useEdgeStore();
  const isConnected = isInputConnected(nodeId, fieldKey);
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
    const mockSchema = {
      type: typeOption.type,
      format: typeOption.format,
      secret: typeOption.secret,
      enum: typeOption.enum,
    };

    const inputType = mapJsonSchemaTypeToInputType(mockSchema as RJSFSchema);

    if (inputType === InputType.OBJECT_EDITOR) {
      const nestedFieldKey = generateHandleId(idSchema.$id ?? "");
      return (
        <InputRenderer
          id={`${name}-input`}
          type={inputType}
          required={false}
          nodeId={nodeId}
          fieldKey={nestedFieldKey}
          value={formData}
          onChange={handleValueChange}
        />
      );
    }

    return (
      <InputRenderer
        type={inputType}
        id={`${name}-input`}
        value={formData ?? (inputType === InputType.NUMBER ? "" : "")}
        placeholder={`Enter ${name}`}
        required={false}
        onChange={handleValueChange}
      />
    );
  };

  if (isNullableType && nonNull) {
    return (
      <div className="flex flex-col gap-2">
        <div className="flex items-center justify-between gap-2">
          <div className="-ml-2 flex items-center gap-1">
            <NodeHandle id={fieldKey} isConnected={isConnected} side="left" />

            <Text variant="body">
              {name.charAt(0).toUpperCase() + name.slice(1)}
            </Text>
            <Text variant="small" className="!text-green-500">
              ({nonNull.type} | null)
            </Text>
          </div>
          {!isConnected && (
            <Switch
              checked={isEnabled}
              onCheckedChange={handleNullableToggle}
            />
          )}
        </div>
        {!isConnected && isEnabled && renderInput(nonNull)}
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-2">
      <div className="-ml-2 flex items-center gap-1">
        <NodeHandle id={fieldKey} isConnected={isConnected} side="left" />
        <Text variant="body">
          {name.charAt(0).toUpperCase() + name.slice(1)}
        </Text>
        {!isConnected && (
          <Select
            label=""
            id={`${name}-type-select`}
            hideLabel={true}
            value={selectedType}
            onValueChange={handleTypeChange}
            options={typeOptions.map((o) => ({ value: o.type, label: o.type }))}
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
