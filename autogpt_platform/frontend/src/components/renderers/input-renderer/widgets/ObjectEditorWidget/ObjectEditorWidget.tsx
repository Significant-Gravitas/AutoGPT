"use client";

import React from "react";
import { Plus, X } from "lucide-react";
import { Text } from "@/components/atoms/Text/Text";
import { Button } from "@/components/atoms/Button/Button";
import { Input } from "@/components/atoms/Input/Input";
import NodeHandle from "@/app/(platform)/build/components/FlowEditor/handlers/NodeHandle";
import { useEdgeStore } from "@/app/(platform)/build/stores/edgeStore";
import {
  generateHandleId,
  HandleIdType,
  parseKeyValueHandleId,
} from "@/app/(platform)/build/components/FlowEditor/handlers/helpers";
import { cn } from "@/lib/utils";

export interface ObjectEditorProps {
  id: string;
  value?: Record<string, any>;
  onChange?: (value: Record<string, any>) => void;
  placeholder?: string;
  disabled?: boolean;
  className?: string;
  nodeId: string;
  fieldKey: string;
  brokenInputs?: Set<string>;
  typeMismatchInputs?: Map<string, string>; // Map of input name -> new type
}

export const ObjectEditor = React.forwardRef<HTMLDivElement, ObjectEditorProps>(
  (
    {
      id: parentFieldId,
      value = {},
      onChange,
      placeholder = "Enter value",
      disabled = false,
      className,
      nodeId,
      brokenInputs,
      typeMismatchInputs,
    },
    ref,
  ) => {
    const getAllHandleIdsOfANode = useEdgeStore(
      (state) => state.getAllHandleIdsOfANode,
    );
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

    const allHandleIdsOfANode = getAllHandleIdsOfANode(nodeId);
    const allKeyValueHandleIdsOfANode = allHandleIdsOfANode.filter((handleId) =>
      handleId.includes("_#_"),
    );
    allKeyValueHandleIdsOfANode.forEach((handleId) => {
      const key = parseKeyValueHandleId(handleId, HandleIdType.KEY_VALUE);
      if (!value[key]) {
        value[key] = null;
      }
    });

    // Note: ObjectEditor is always used in node context, so showHandles is always true
    // If you need to use it in dialog context, you'll need to pass showHandles via props
    const showHandles = true;

    return (
      <div
        ref={ref}
        className={cn("flex flex-col gap-2", className)}
        id={parentFieldId}
      >
        {Object.entries(value).map(([key, propertyValue], idx) => {
          const handleId = generateHandleId(
            parentFieldId,
            [key],
            HandleIdType.KEY_VALUE,
          );
          const isDynamicPropertyConnected = isInputConnected(nodeId, handleId);
          const isBroken = handleId
            ? (brokenInputs?.has(handleId) ?? false)
            : false;
          const newTypeFromMismatch = handleId
            ? typeMismatchInputs?.get(handleId)
            : undefined;
          const hasTypeMismatch = !!newTypeFromMismatch;

          return (
            <div key={idx} className="flex flex-col gap-2">
              <div className="-ml-2 flex items-center gap-1">
                {showHandles && (
                  <NodeHandle
                    isConnected={isDynamicPropertyConnected}
                    handleId={handleId}
                    side="left"
                    isBroken={isBroken}
                  />
                )}
                <Text
                  variant="small"
                  className={cn(
                    "!text-gray-500",
                    isBroken && "!text-red-500 line-through",
                  )}
                >
                  #{key.trim() === "" ? "" : key}
                </Text>
                <Text
                  variant="small"
                  className={cn(
                    "!text-green-500",
                    isBroken && "!text-red-500 line-through",
                    hasTypeMismatch && "rounded bg-red-100 px-1 !text-red-600",
                  )}
                >
                  ({hasTypeMismatch ? newTypeFromMismatch : "string"})
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
                    value={propertyValue ?? ""}
                    onChange={(e) => setProperty(key, e.target.value)}
                    placeholder={placeholder}
                    wrapperClassName="mb-0"
                    disabled={disabled}
                  />
                  <Button
                    type="button"
                    variant="secondary"
                    size="small"
                    className="min-w-10"
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
          variant="secondary"
          size="small"
          onClick={addProperty}
          className="w-fit"
          disabled={hasEmptyKeys || disabled}
        >
          <Plus className="h-4 w-4" />
          Add Property
        </Button>
      </div>
    );
  },
);

ObjectEditor.displayName = "ObjectEditor";
