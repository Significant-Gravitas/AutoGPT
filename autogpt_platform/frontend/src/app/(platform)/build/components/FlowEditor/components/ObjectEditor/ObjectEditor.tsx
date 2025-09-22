"use client";

import React from "react";
import { Plus, X } from "lucide-react";
import { Text } from "@/components/atoms/Text/Text";
import { Button } from "@/components/atoms/Button/Button";
import { Input } from "@/components/atoms/Input/Input";
import NodeHandle from "../../handlers/NodeHandle";
import { useEdgeStore } from "@/app/(platform)/build/stores/edgeStore";
import { generateHandleId, HandleIdType } from "../../handlers/helpers";

export interface ObjectEditorProps {
  id: string;
  value?: Record<string, any>;
  onChange?: (value: Record<string, any>) => void;
  placeholder?: string;
  disabled?: boolean;
  className?: string;
  nodeId: string;
  fieldKey: string;
}

export const ObjectEditor = React.forwardRef<HTMLDivElement, ObjectEditorProps>(
  (
    {
      id,
      value = {},
      onChange,
      placeholder = "Enter value",
      disabled = false,
      className,
      nodeId,
      fieldKey,
    },
    ref,
  ) => {
    const setProperty = (key: string, propertyValue: any) => {
      if (!onChange) return;

      const newData: Record<string, any> = { ...value };
      if (propertyValue === undefined || propertyValue === "") {
        delete newData[key];
      } else {
        newData[key] = propertyValue;
      }
      onChange(newData);
    };

    const addProperty = () => {
      if (!onChange) return;
      onChange({ ...value, [""]: "" });
    };

    const removeProperty = (key: string) => {
      if (!onChange) return;
      const newData = { ...value };
      delete newData[key];
      onChange(newData);
    };

    const updateKey = (oldKey: string, newKey: string) => {
      if (!onChange || oldKey === newKey) return;

      const propertyValue = value[oldKey];
      const newData: Record<string, any> = { ...value };
      delete newData[oldKey];
      newData[newKey] = propertyValue;
      onChange(newData);
    };

    const hasEmptyKeys = Object.keys(value).some((key) => key.trim() === "");

    const { isInputConnected } = useEdgeStore();

    return (
      <div
        ref={ref}
        className={`flex flex-col gap-2 ${className || ""}`}
        id={id}
      >
        {Object.entries(value).map(([key, propertyValue], idx) => {
          const dynamicHandleId = generateHandleId(
            fieldKey,
            [key],
            HandleIdType.KEY_VALUE,
          );
          const isDynamicPropertyConnected = isInputConnected(
            nodeId,
            dynamicHandleId,
          );

          console.log("dynamicHandleId", dynamicHandleId);
          console.log("key", key);
          console.log("fieldKey", fieldKey);

          return (
            <div key={idx} className="flex flex-col gap-2">
              <div className="-ml-2 flex items-center gap-1">
                <NodeHandle
                  id={dynamicHandleId}
                  isConnected={isDynamicPropertyConnected}
                  side="left"
                />

                <Text variant="small" className="text-gray-600">
                  #{key.trim() === "" ? "" : key}
                </Text>
                <Text variant="small" className="!text-green-500">
                  (string)
                </Text>
              </div>
              {!isDynamicPropertyConnected && (
                <div className="flex items-center gap-2">
                  <Input
                    hideLabel={true}
                    label=""
                    id={`key-${idx}`}
                    size="small"
                    value={key}
                    onChange={(e) => updateKey(key, e.target.value)}
                    placeholder="Key"
                    wrapperClassName="mb-0"
                    disabled={disabled}
                  />
                  <Input
                    hideLabel={true}
                    label=""
                    id={`value-${idx}`}
                    size="small"
                    value={propertyValue as string}
                    onChange={(e) => setProperty(key, e.target.value)}
                    placeholder={placeholder}
                    wrapperClassName="mb-0"
                    disabled={disabled}
                  />
                  <Button
                    type="button"
                    variant="secondary"
                    size="small"
                    onClick={() => removeProperty(key)}
                    disabled={disabled}
                  >
                    <X className="h-4 w-4" />
                  </Button>
                </div>
              )}
            </div>
          );
        })}

        <Button
          type="button"
          size="small"
          onClick={addProperty}
          className="w-full"
          disabled={hasEmptyKeys || disabled}
        >
          <Plus className="mr-2 h-4 w-4" />
          Add Property
        </Button>
      </div>
    );
  },
);

ObjectEditor.displayName = "ObjectEditor";
