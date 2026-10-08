"use client";

import { Text } from "@/components/atoms/Text/Text";
import {
  RadioGroup as KobraRadioGroup,
  RadioGroupItem,
} from "@/components/ui/radio-group";
import { cn } from "@/lib/utils";
import { useId } from "react";

export interface RadioGroupOption {
  value: string;
  label: string;
  description?: string;
  disabled?: boolean;
}

interface Props {
  label: string;
  hideLabel?: boolean;
  options: RadioGroupOption[];
  value: string;
  onValueChange: (value: string) => void;
  disabled?: boolean;
  className?: string;
}

export function RadioGroup({
  label,
  hideLabel = false,
  options,
  value,
  onValueChange,
  disabled,
  className,
}: Props) {
  const labelId = useId();

  return (
    <div className={cn("flex w-full flex-col gap-3", className)}>
      <Text
        variant="body-medium"
        as="span"
        id={labelId}
        className={cn(hideLabel && "sr-only")}
      >
        {label}
      </Text>
      <KobraRadioGroup
        aria-labelledby={labelId}
        value={value}
        onValueChange={(next) => onValueChange(String(next))}
        disabled={disabled}
        className="gap-3"
      >
        {options.map((option) => (
          <label
            key={option.value}
            className="flex cursor-pointer items-start gap-3 rounded-xl border border-border bg-card px-4 py-3 transition-colors duration-150 hover:bg-zinc-50 has-data-checked:border-zinc-800 has-data-disabled:cursor-not-allowed has-data-disabled:opacity-60"
          >
            <RadioGroupItem
              value={option.value}
              disabled={option.disabled}
              className="mt-0.5"
            />
            <span className="flex min-w-0 flex-col gap-0.5">
              <Text variant="body-medium" as="span">
                {option.label}
              </Text>
              {option.description ? (
                <Text variant="body" as="span" tone="muted">
                  {option.description}
                </Text>
              ) : null}
            </span>
          </label>
        ))}
      </KobraRadioGroup>
    </div>
  );
}
