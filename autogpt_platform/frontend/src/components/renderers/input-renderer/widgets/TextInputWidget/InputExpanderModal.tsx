"use client";

import React, { FC, useEffect, useState } from "react";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { CheckIcon, CopyIcon } from "@phosphor-icons/react";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { cn } from "@/lib/utils";
import { Input } from "@/components/atoms/Input/Input";

interface InputExpanderModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSave: (value: string) => void;
  title?: string;
  defaultValue: string;
  description?: string;
  placeholder?: string;
}

export const InputExpanderModal: FC<InputExpanderModalProps> = ({
  isOpen,
  onClose,
  onSave,
  title,
  defaultValue,
  description,
  placeholder,
}) => {
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
          <Text variant="h4" className="text-slate-900">
            {title || "Edit Text"}
          </Text>
          <Text variant="body">{description}</Text>
          <Input
            type="textarea"
            label=""
            hideLabel
            id="input-expander-modal"
            value={tempValue}
            className="!min-h-[300px] rounded-2xlarge"
            onChange={(e) => setTempValue(e.target.value)}
            placeholder={placeholder || "Enter text..."}
            autoFocus
          />

          <div className="flex items-center justify-end gap-1">
            <Button
              variant="secondary"
              size="small"
              onClick={copyValue}
              className={cn(
                "h-fit min-w-0 gap-1.5 border border-zinc-200 p-2 text-black hover:text-slate-900",
                isCopied &&
                  "border-green-400 bg-green-100 hover:border-green-400 hover:bg-green-200",
              )}
            >
              {isCopied ? (
                <CheckIcon size={16} className="text-green-600" />
              ) : (
                <CopyIcon size={16} />
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
