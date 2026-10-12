"use client";

import { useId } from "react";

import { Switch } from "@/components/atoms/Switch/Switch";
import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";

interface Props {
  label: string;
  description: string;
  checked: boolean;
  disabled?: boolean;
  describedBy?: string;
  onCheckedChange: () => void;
}

export function SettingRow({
  label,
  description,
  checked,
  disabled,
  describedBy,
  onCheckedChange,
}: Props) {
  const switchId = useId();
  const descriptionId = useId();

  return (
    <div className="flex items-center justify-between gap-4">
      <div className={cn("flex flex-col", disabled && "opacity-50")}>
        <label
          htmlFor={switchId}
          className={disabled ? "cursor-not-allowed" : "cursor-pointer"}
        >
          <Text variant="body-medium" as="span" className="text-textBlack">
            {label}
          </Text>
        </label>
        <Text
          id={descriptionId}
          variant="small"
          as="span"
          className="text-zinc-500"
        >
          {description}
        </Text>
      </div>
      <Switch
        id={switchId}
        checked={checked}
        disabled={disabled}
        onCheckedChange={onCheckedChange}
        aria-describedby={[descriptionId, describedBy]
          .filter(Boolean)
          .join(" ")}
      />
    </div>
  );
}
