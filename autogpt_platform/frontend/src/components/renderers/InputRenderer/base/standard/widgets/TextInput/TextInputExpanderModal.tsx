"use client";

import React, { FC, useEffect, useId, useState } from "react";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { cn } from "@/lib/utils";
import { Input } from "@/components/atoms/Input/Input";
import { Copy01Icon, Tick02Icon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";

interface InputExpanderModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSave: (value: string) => void;
  title?: string;
  defaultValue: string;
  description?: string;
  placeholder?: string;
  inputType?: "text" | "json";
}

export const InputExpanderModal: FC<InputExpanderModalProps> = ({
  isOpen,
  onClose,
  onSave,
  title,
  defaultValue,
  description,
  placeholder,
  inputType = "text",
}) => {
  const inputId = useId();
  const [tempValue, setTempValue] = useState(defaultValue);
  const [isCopied, setIsCopied] = useState(false);
  const { toast } = useToast();

  useEffect(() => {
    if (isOpen) {
      setTempValue(defaultValue);
      setIsCopied(false);
    }
  }, [isOpen, defaultValue]);

  const handleSave = () => {
    onSave(tempValue);
    onClose();
  };

  const copyValue = () => {
    navigator.clipboard.writeText(tempValue).then(() => {
      setIsCopied(true);
      toast({
        title: "Copied to clipboard!",
        duration: 2000,
      });
      setTimeout(() => setIsCopied(false), 2000);
    });
  };

  return (
    <Dialog
      title={title || "Edit Text"}
      controlled={{
        isOpen,
        set: async (open) => {
          if (!open) onClose();
        },
      }}
      onClose={onClose}
      styling={{ maxWidth: "600px", minWidth: "600px" }}
    >
      <Dialog.Content>
        <div className="flex flex-col gap-4 px-1">
          <Text variant="body">{description}</Text>
          <Input
            type="textarea"
            label={title || "Edit value"}
            hideLabel
            id={inputId}
            value={tempValue}
            className={cn(
              "!min-h-[300px]",
              inputType === "json" && "font-mono text-sm",
            )}
            onChange={(e) => setTempValue(e.target.value)}
            placeholder={placeholder || "Enter text..."}
            autoFocus
          />

          <div className="flex items-center justify-end gap-1">
            <Button
              variant="secondary"
              size="small"
              onClick={copyValue}
              aria-label={
                isCopied ? "Copied to clipboard" : "Copy to clipboard"
              }
              className={cn(
                // TODO(tokens): no exact role for hover:text-slate-900 #2A313A
                "h-fit min-w-0 gap-1.5 border border-border-base p-2 text-foreground-base hover:text-slate-900",
                isCopied &&
                  // TODO(tokens): no exact role for border-green-400 #45b56e, bg-green-100 #b7e2c7, hover:border-green-400 #45b56e, hover:bg-green-200 #94d5ac
                  "border-green-400 bg-green-100 hover:border-green-400 hover:bg-green-200",
              )}
            >
              {isCopied ? (
                <Icon icon={Tick02Icon} size={16} className="text-success" />
              ) : (
                <Icon icon={Copy01Icon} size={16} />
              )}
            </Button>
          </div>

          <Dialog.Footer>
            <Button variant="secondary" size="small" onClick={onClose}>
              Cancel
            </Button>
            <Button variant="primary" size="small" onClick={handleSave}>
              Save
            </Button>
          </Dialog.Footer>
        </div>
      </Dialog.Content>
    </Dialog>
  );
};
