import React from "react";
import { FieldProps } from "@rjsf/utils";
import { LocalValuedInput } from "@/components/ui/input";

import { Plus, X } from "lucide-react";
import { Text } from "@/components/atoms/Text/Text";
import { getDefaultRegistry } from "@rjsf/core";
import { Button } from "@/components/atoms/Button/Button";
import { Input } from "@/components/atoms/Input/Input";

export const ObjectField = (props: FieldProps) => {
  const { schema, formData = {}, onChange, name, idSchema } = props;
  const DefaultObjectField = getDefaultRegistry().fields.ObjectField;

  // Let the default field render for root or fixed-schema objects
  const isFreeForm =
    !schema.properties ||
    Object.keys(schema.properties).length === 0 ||
    schema.additionalProperties === true;

  if (idSchema?.$id === "root" || !isFreeForm) {
    return <DefaultObjectField {...props} />;
  }

  const setProperty = (key: string, value: any) => {
    const newData: Record<string, any> = { ...formData };
    if (value === undefined || value === "") {
      delete newData[key];
    } else {
      newData[key] = value;
    }
    onChange(newData);
  };

  const addProperty = () => {
    onChange({ ...formData, [""]: "" });
  };

  const removeProperty = (key: string) => {
    const newData = { ...formData };
    delete newData[key];
    onChange(newData);
  };

  const updateKey = (oldKey: string, newKey: string) => {
    if (oldKey === newKey) return;
    const value = (formData as Record<string, any>)[oldKey];
    const newData: Record<string, any> = { ...formData };
    delete newData[oldKey];
    newData[newKey] = value;
    onChange(newData);
  };

  const hasEmptyKeys = Object.keys(formData).some((key) => key.trim() === "");

  return (
    <div className="flex flex-col gap-2">
      {Object.entries(formData).map(([key, value], idx) => (
        <div key={idx} className="flex flex-col gap-2">
          <div className="flex items-center gap-1">
            <Text variant="small" className="text-gray-600">
              #{key.trim() === "" ? "(text)" : key}
            </Text>
            <Text variant="small" className="text-green-500">
              (text)
            </Text>
          </div>

          <div className="flex items-center gap-2">
            <Input
              hideLabel={true}
              label={""}
              id={key}
              size="small"
              value={key}
              onChange={(e) => updateKey(key, e.target.value)}
              placeholder="Key"
              wrapperClassName="mb-0"
            />
            <Input
              hideLabel={true}
              label={""}
              id={key}
              size="small"
              value={value as string}
              onChange={(e) => setProperty(key, e.target.value)}
              placeholder={`Enter ${name || "Contact Data"}`}
              wrapperClassName="mb-0"
            />
            <Button
              type="button"
              variant="secondary"
              size="small"
              onClick={() => removeProperty(key)}
            >
              <X className="h-4 w-4" />
            </Button>
          </div>
        </div>
      ))}

      <Button
        type="button"
        size="small"
        onClick={addProperty}
        className="w-full"
        disabled={hasEmptyKeys}
      >
        <Plus className="mr-2 h-4 w-4" />
        Add Property
      </Button>
    </div>
  );
};
