import React, { useState, useEffect } from "react";
import { FieldProps } from "@rjsf/utils";

import { LocalValuedInput } from "@/components/ui/input";
import { Text } from "@/components/atoms/Text/Text";
import { Switch } from "@/components/atoms/Switch/Switch";
import { Input } from "@/components/atoms/Input/Input";
import { Select } from "@/components/atoms/Select/Select";
import { DateInputWidget } from "../widgets/DateInputWidget";

// Create your custom AnyOf field
export const AnyOfField = (props: FieldProps) => {
  const { schema, formData, onChange, name } = props;

  // Extract types from anyOf schema
  const typeOptions =
    schema.anyOf?.map((option: any, index: number) => ({
      type: option.type || "string",
      title: option.title || `Option ${index + 1}`,
      index,
    })) || [];

  // Check if this is a nullable type (e.g., string | null, number | null)
  const isNullableType =
    typeOptions.length === 2 &&
    typeOptions.some((opt) => opt.type === "null") &&
    typeOptions.some((opt) => opt.type !== "null");

  // Get the non-null type for nullable fields
  const nonNullType = isNullableType
    ? typeOptions.find((opt) => opt.type !== "null")?.type
    : null;

  // State for nullable fields
  const [isEnabled, setIsEnabled] = useState<boolean>(
    formData !== null && formData !== undefined,
  );

  // Default to first type if available (for non-nullable fields)
  const [selectedType, setSelectedType] = useState<any>(
    isNullableType ? nonNullType : schema.default || typeOptions[0]?.type,
  );

  // Initialize with first type on mount
  useEffect(() => {
    if (!isNullableType && typeOptions.length > 0 && !formData) {
      setSelectedType(typeOptions[0].type);
    }
  }, [typeOptions, formData, isNullableType]);

  // Update enabled state when formData changes (for nullable fields)
  useEffect(() => {
    if (isNullableType) {
      setIsEnabled(formData !== null && formData !== undefined);
    }
  }, [formData, isNullableType]);

  const handleTypeChange = (newType: string) => {
    setSelectedType(newType);
    // Clear the current value when type changes
    onChange(undefined);
  };

  const handleValueChange = (value: any) => {
    onChange(value);
  };

  // Handle nullable field switch change
  const handleNullableToggle = (checked: boolean) => {
    setIsEnabled(checked);
    if (!checked) {
      // Set to null when unchecked
      onChange(null);
    } else {
      // Clear value when enabling (let user input new value)
      onChange(undefined);
    }
  };

  const renderInputForType = (
    inputType: string = selectedType,
    format?: string,
  ) => {
    switch (inputType) {
      case "string":
        return (
          <Input
            hideLabel={true}
            label={""}
            id={`${name}-input`}
            size="small"
            wrapperClassName="mb-0"
            type="text"
            value={formData || ""}
            onChange={(e) => handleValueChange(e.target.value)}
            placeholder={`Enter ${inputType} value`}
            className="w-full"
          />
        );
      case "number":
        return (
          <Input
            hideLabel={true}
            label={""}
            id={`${name}-input`}
            size="small"
            wrapperClassName="mb-0"
            type="number"
            value={formData || ""}
            onChange={(e) => handleValueChange(Number(e.target.value))}
            placeholder={`Enter ${inputType} value`}
            className="w-full"
          />
        );
      case "boolean":
        return (
          <Switch
            checked={formData}
            onCheckedChange={(value) => handleValueChange(value)}
          />
        );
      default:
        return (
          <LocalValuedInput
            type="text"
            value={formData || ""}
            onChange={(e) => handleValueChange(e.target.value)}
            placeholder={`Enter ${inputType} value`}
            className="w-full"
          />
        );
    }
  };

  // Render nullable type UI
  if (isNullableType) {
    return (
      <div className="flex flex-col gap-2">
        <div className="flex items-center justify-between gap-2">
          <div className="flex items-center gap-1">
            <Text variant="body">
              {name.charAt(0).toUpperCase() + name.slice(1)}
            </Text>
            <Text variant="small" className="!text-green-500">
              ({nonNullType} | null)
            </Text>
          </div>
          <Switch checked={isEnabled} onCheckedChange={handleNullableToggle} />
        </div>
        {isEnabled && renderInputForType(nonNullType, schema.format)}
      </div>
    );
  }

  // Render regular select dropdown UI for non-nullable types
  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-center gap-1">
        <Text variant="body">
          {name.charAt(0).toUpperCase() + name.slice(1)}
        </Text>
        <Select
          label=""
          id={`${name}-type-select`}
          hideLabel={true}
          value={selectedType}
          onValueChange={handleTypeChange}
          options={typeOptions.map((option) => ({
            value: option.type,
            label: option.type,
          }))}
          size="small"
          wrapperClassName="!mb-0 "
          className="h-6 w-fit gap-1 pl-3 pr-2"
        />
      </div>
      {renderInputForType()}
    </div>
  );
};
