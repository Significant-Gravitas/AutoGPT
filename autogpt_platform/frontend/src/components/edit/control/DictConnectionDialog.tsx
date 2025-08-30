import React, { useState, useCallback } from "react";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { getTypeTextColor } from "@/lib/utils";

interface DictConnectionDialogProps {
  isOpen: boolean;
  onClose: () => void;
  onConfirm: (
    key: string,
    connectToValue: boolean,
    staticValue?: string,
  ) => void;
  title: string;
  description?: string;
  keyType?: string;
  valueType?: string;
  sourceType?: string; // The type we're connecting from
}

export function DictConnectionDialog({
  isOpen,
  onClose,
  onConfirm,
  title,
  description,
  keyType = "string",
  valueType,
  sourceType,
}: DictConnectionDialogProps) {
  const [keyName, setKeyName] = useState("");
  const [connectTo, setConnectTo] = useState<"key" | "value">("value");
  const [staticValue, setStaticValue] = useState("");

  // Determine which option should be available based on type compatibility
  const canConnectToKey =
    sourceType === keyType || keyType === "any" || sourceType === "any";
  const canConnectToValue =
    sourceType === valueType || valueType === "any" || sourceType === "any";

  // Auto-select if only one option is available
  React.useEffect(() => {
    if (canConnectToValue && !canConnectToKey) {
      setConnectTo("value");
    } else if (canConnectToKey && !canConnectToValue) {
      setConnectTo("key");
    }
  }, [canConnectToKey, canConnectToValue]);

  const handleConfirm = useCallback(() => {
    if (keyName.trim()) {
      const needsStaticValue =
        connectTo === "key" ? valueType !== "any" : keyType !== "any";
      onConfirm(
        keyName.trim(),
        connectTo === "value",
        needsStaticValue && staticValue ? staticValue : undefined,
      );
      onClose();
    }
  }, [keyName, connectTo, staticValue, keyType, valueType, onConfirm, onClose]);

  const TYPE_NAME: Record<string, string> = {
    string: "text",
    number: "number",
    integer: "integer",
    boolean: "true/false",
    object: "object",
    array: "list",
    null: "null",
    any: "any",
  };

  return (
    <Dialog open={isOpen} onOpenChange={(open) => !open && onClose()}>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle>{title || "Add Dictionary Entry"}</DialogTitle>
          {description && <DialogDescription>{description}</DialogDescription>}
        </DialogHeader>
        <div className="space-y-4">
          {/* Key Name Input */}
          <div>
            <Label htmlFor="key-name">
              Dictionary Key Name
              <span className={`ml-2 text-sm ${getTypeTextColor(keyType)}`}>
                ({TYPE_NAME[keyType] || keyType})
              </span>
            </Label>
            <Input
              id="key-name"
              type="text"
              placeholder="Enter key name"
              value={keyName}
              onChange={(e) => setKeyName(e.target.value)}
              className="mt-2"
              autoFocus
            />
          </div>

          {/* Connection Target Selection */}
          {canConnectToKey && canConnectToValue && (
            <div>
              <Label>Connect to:</Label>
              <RadioGroup
                value={connectTo}
                onValueChange={(value) =>
                  setConnectTo(value as "key" | "value")
                }
                className="mt-2 space-y-2"
              >
                <div className="flex items-center space-x-2">
                  <RadioGroupItem
                    value="key"
                    id="connect-key"
                    disabled={!canConnectToKey}
                  />
                  <Label htmlFor="connect-key" className="cursor-pointer">
                    Key
                    <span
                      className={`ml-2 text-sm ${getTypeTextColor(keyType)}`}
                    >
                      ({TYPE_NAME[keyType] || keyType})
                    </span>
                    {sourceType &&
                      sourceType !== keyType &&
                      keyType !== "any" && (
                        <span className="ml-2 text-sm text-orange-500">
                          (type mismatch: {sourceType} → {keyType})
                        </span>
                      )}
                  </Label>
                </div>
                <div className="flex items-center space-x-2">
                  <RadioGroupItem
                    value="value"
                    id="connect-value"
                    disabled={!canConnectToValue}
                  />
                  <Label htmlFor="connect-value" className="cursor-pointer">
                    Value
                    <span
                      className={`ml-2 text-sm ${getTypeTextColor(valueType || "any")}`}
                    >
                      ({TYPE_NAME[valueType || "any"] || valueType})
                    </span>
                    {sourceType &&
                      sourceType !== valueType &&
                      valueType !== "any" && (
                        <span className="ml-2 text-sm text-orange-500">
                          (type mismatch: {sourceType} → {valueType})
                        </span>
                      )}
                  </Label>
                </div>
              </RadioGroup>
            </div>
          )}

          {/* Static Value Input for the non-connected part */}
          {((connectTo === "key" && valueType && valueType !== "any") ||
            (connectTo === "value" && keyType !== "string")) && (
            <div>
              <Label htmlFor="static-value">
                {connectTo === "key" ? "Value" : "Key"} (will be set to):
                <span
                  className={`ml-2 text-sm ${getTypeTextColor(connectTo === "key" ? valueType || "any" : keyType)}`}
                >
                  (
                  {TYPE_NAME[
                    connectTo === "key" ? valueType || "any" : keyType
                  ] || (connectTo === "key" ? valueType : keyType)}
                  )
                </span>
              </Label>
              <Input
                id="static-value"
                type={
                  connectTo === "value" && keyType === "number"
                    ? "number"
                    : "text"
                }
                placeholder={`Enter ${connectTo === "key" ? "value" : "key"}`}
                value={staticValue}
                onChange={(e) => setStaticValue(e.target.value)}
                className="mt-2"
              />
              {connectTo === "value" && keyType === "string" && (
                <p className="mt-1 text-sm text-gray-500">
                  If left empty, the key name will be used
                </p>
              )}
            </div>
          )}

          <div className="flex justify-end space-x-2">
            <Button variant="outline" onClick={onClose}>
              Cancel
            </Button>
            <Button onClick={handleConfirm} disabled={!keyName.trim()}>
              Add Connection
            </Button>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}
