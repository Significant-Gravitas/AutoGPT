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
import { beautifyString, getTypeTextColor } from "@/lib/utils";
import { BlockIOSubSchema } from "@/lib/autogpt-server-api";

interface ConnectionOption {
  handleId: string;
  schema: BlockIOSubSchema;
  isRequired?: boolean;
  allowDynamicKey?: boolean;
  dynamicKeyType?: string;
}

interface ConnectionSelectorProps {
  isOpen: boolean;
  onClose: () => void;
  onSelect: (handleId: string, dynamicKey?: string) => void;
  options: ConnectionOption[];
  title: string;
  description?: string;
  allowDynamicKey?: boolean;
  dynamicKeyType?: string;
}

export function ConnectionSelector({
  isOpen,
  onClose,
  onSelect,
  options,
  title,
  description,
}: ConnectionSelectorProps) {
  const [selectedOption, setSelectedOption] = useState<string>(
    options[0]?.handleId || "",
  );
  const [useDynamicKey, setUseDynamicKey] = useState(false);
  const [dynamicKey, setDynamicKey] = useState("");

  // Find the currently selected option to check if it supports dynamic keys
  const currentOption = options.find((opt) => opt.handleId === selectedOption);

  // Validate that the option actually supports dynamic connections
  const supportsDynamic =
    currentOption?.allowDynamicKey &&
    (currentOption.schema.type === "array" ||
      (currentOption.schema.type === "object" &&
        "additionalProperties" in currentOption.schema &&
        currentOption.schema.additionalProperties));

  const handleConfirm = useCallback(() => {
    const currentOpt = options.find((opt) => opt.handleId === selectedOption);

    if (useDynamicKey) {
      if (currentOpt?.schema.type === "array") {
        // For arrays, pass a special marker to indicate dynamic append
        // We use '__append__' as a special dynamic key to signal array append
        onSelect(selectedOption, "__append__");
      } else if (
        dynamicKey &&
        currentOpt?.schema.type === "object" &&
        "additionalProperties" in currentOpt.schema &&
        currentOpt.schema.additionalProperties
      ) {
        // For dicts with additionalProperties, pass the key with dot notation
        onSelect(selectedOption, dynamicKey);
      } else {
        // Not a dynamic type or dict without key - just select normally
        if (!dynamicKey) return; // Need a key for dict
        onSelect(selectedOption); // Regular selection
      }
    } else if (selectedOption) {
      onSelect(selectedOption);
    }
    onClose();
  }, [selectedOption, useDynamicKey, dynamicKey, options, onSelect, onClose]);

  const TYPE_NAME: Record<string, string> = {
    string: "text",
    number: "number",
    integer: "integer",
    boolean: "true/false",
    object: "object",
    array: "list",
    null: "null",
  };

  return (
    <Dialog open={isOpen} onOpenChange={(open) => !open && onClose()}>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle>{title}</DialogTitle>
          {description && <DialogDescription>{description}</DialogDescription>}
        </DialogHeader>
        <div className="space-y-4">
          <div className="max-h-[40vh] overflow-y-auto">
            <RadioGroup
              value={selectedOption}
              onValueChange={setSelectedOption}
              className="space-y-2 p-1"
            >
              {options.map((option) => (
                <Label
                  key={option.handleId}
                  htmlFor={option.handleId}
                  className="flex cursor-pointer items-center space-x-2 rounded-lg border p-3 hover:bg-gray-50 dark:border-slate-700 dark:hover:bg-slate-800"
                  onClick={() => setSelectedOption(option.handleId)}
                >
                  <RadioGroupItem
                    value={option.handleId}
                    id={option.handleId}
                  />
                  <div className="flex-1">
                    <div className="flex items-center justify-between">
                      <span className="font-medium">
                        {option.schema.title ||
                          beautifyString(option.handleId.toLowerCase())}
                        {option.isRequired && "*"}
                      </span>
                      <span
                        className={`text-sm ${getTypeTextColor(
                          option.schema.type || "any",
                        )}`}
                      >
                        (
                        {TYPE_NAME[
                          option.schema.type as keyof typeof TYPE_NAME
                        ] || "any"}
                        )
                      </span>
                    </div>
                    {option.schema.description && (
                      <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
                        {option.schema.description}
                      </p>
                    )}
                  </div>
                </Label>
              ))}
            </RadioGroup>
          </div>

          {supportsDynamic && (
            <div className="space-y-2 border-t pt-2">
              <Label
                htmlFor="dynamic-key"
                className="flex cursor-pointer items-center space-x-2 rounded p-2 hover:bg-gray-50 dark:hover:bg-slate-800"
                onClick={() => setUseDynamicKey(!useDynamicKey)}
              >
                <input
                  type="checkbox"
                  id="dynamic-key"
                  checked={useDynamicKey}
                  onChange={(e) => setUseDynamicKey(e.target.checked)}
                  className="h-4 w-4"
                  onClick={(e) => e.stopPropagation()}
                />
                <span>
                  {currentOption.schema.type === "array"
                    ? "Append to list"
                    : "Add as new dictionary key"}
                  {currentOption.dynamicKeyType && (
                    <span
                      className={`ml-2 text-sm ${getTypeTextColor(currentOption.dynamicKeyType)}`}
                    >
                      (
                      {TYPE_NAME[
                        currentOption.dynamicKeyType as keyof typeof TYPE_NAME
                      ] || currentOption.dynamicKeyType}
                      )
                    </span>
                  )}
                </span>
              </Label>
              {useDynamicKey && currentOption.schema.type !== "array" && (
                <Input
                  type="text"
                  placeholder="Enter dictionary key"
                  value={dynamicKey}
                  onChange={(e) => setDynamicKey(e.target.value)}
                  className="mt-2"
                />
              )}
            </div>
          )}

          <div className="flex justify-end space-x-2">
            <Button variant="outline" onClick={onClose}>
              Cancel
            </Button>
            <Button
              onClick={handleConfirm}
              disabled={
                !selectedOption ||
                (useDynamicKey &&
                  currentOption?.schema.type !== "array" &&
                  !dynamicKey.trim())
              }
            >
              Connect
            </Button>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}

interface DynamicKeyDialogProps {
  isOpen: boolean;
  onClose: () => void;
  onConfirm: (key: string) => void;
  title: string;
  description?: string;
  keyType?: string;
  isArray?: boolean;
}

export function DynamicKeyDialog({
  isOpen,
  onClose,
  onConfirm,
  title,
  description,
  keyType,
  isArray = false,
}: DynamicKeyDialogProps) {
  const [keyValue, setKeyValue] = useState("");

  const handleConfirm = useCallback(() => {
    if (keyValue.trim()) {
      onConfirm(keyValue.trim());
      onClose();
    }
  }, [keyValue, onConfirm, onClose]);

  const TYPE_NAME: Record<string, string> = {
    string: "text",
    number: "number",
    integer: "integer",
    boolean: "true/false",
    object: "object",
    array: "list",
    null: "null",
  };

  return (
    <Dialog open={isOpen} onOpenChange={(open) => !open && onClose()}>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle>{title}</DialogTitle>
          {description && <DialogDescription>{description}</DialogDescription>}
        </DialogHeader>
        <div className="space-y-4">
          <div>
            <Label htmlFor="key-input">
              {isArray ? "Array Index" : "Key Name"}
              {keyType && (
                <span className={`ml-2 text-sm ${getTypeTextColor(keyType)}`}>
                  ({TYPE_NAME[keyType as keyof typeof TYPE_NAME] || keyType})
                </span>
              )}
            </Label>
            <Input
              id="key-input"
              type={isArray ? "number" : "text"}
              placeholder={
                isArray ? "Enter index (e.g., 0, 1, 2)" : "Enter key name"
              }
              value={keyValue}
              onChange={(e) => setKeyValue(e.target.value)}
              className="mt-2"
              autoFocus
            />
          </div>
          <div className="flex justify-end space-x-2">
            <Button variant="outline" onClick={onClose}>
              Cancel
            </Button>
            <Button onClick={handleConfirm} disabled={!keyValue.trim()}>
              Add
            </Button>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}
