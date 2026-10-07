"use client";

import {
  Select as BaseSelect,
  SelectContent,
  SelectItem,
  SelectSeparator,
  SelectTrigger,
  SelectValue,
} from "@/components/__legacy__/ui/select";
import { cn } from "@/lib/utils";
import * as React from "react";
import { ReactNode, useState } from "react";
import { Text } from "../Text/Text";
import type { Variant } from "../Text/helpers";
import { InformationTooltip } from "@/components/molecules/InformationTooltip/InformationTooltip";
import { fieldVariants } from "../Input/fieldVariants";

export interface SelectOption {
  value: string;
  label: string;
  icon?: ReactNode;
  disabled?: boolean;
  separator?: boolean;
  /** Turns the row into an action: runs on selection instead of changing the value. */
  onSelect?: () => void;
}

export interface SelectFieldProps {
  label: string;
  id: string;
  hideLabel?: boolean;
  error?: string;
  hint?: ReactNode;
  placeholder?: string;
  className?: string;
  disabled?: boolean;
  value?: string;
  onValueChange?: (value: string) => void;
  options: SelectOption[];
  size?: "sm" | "md" | "lg";
  labelVariant?: Variant;
  labelClassName?: string;
  labelTooltip?: string;
  renderItem?: (option: SelectOption) => React.ReactNode;
  wrapperClassName?: string;
  "aria-label"?: string;
  "aria-labelledby"?: string;
  "aria-describedby"?: string;
}

export function Select({
  className,
  label,
  placeholder,
  hideLabel = false,
  hint,
  error,
  id,
  disabled,
  value,
  onValueChange,
  options,
  size = "lg",
  labelVariant = "large-medium",
  labelClassName,
  labelTooltip,
  renderItem,
  wrapperClassName,
  "aria-label": ariaLabel,
  "aria-labelledby": ariaLabelledBy,
  "aria-describedby": ariaDescribedBy,
}: SelectFieldProps) {
  const triggerStyles = cn(
    fieldVariants({ size, invalid: Boolean(error) }),
    "[&[data-placeholder]>span]:font-normal [&[data-placeholder]>span]:text-muted-foreground",
    className,
  );

  const [uncontrolledValue, setUncontrolledValue] = useState<string>();
  const currentValue = value ?? uncontrolledValue ?? "";

  function handleValueChange(nextValue: string) {
    const action = options.find(
      (option) => option.value === nextValue,
    )?.onSelect;
    if (action) {
      action();
      return;
    }
    if (value === undefined) setUncontrolledValue(nextValue);
    onValueChange?.(nextValue);
  }

  const select = (
    <BaseSelect
      value={currentValue}
      onValueChange={handleValueChange}
      disabled={disabled}
    >
      <SelectTrigger
        className={triggerStyles}
        aria-label={ariaLabel ?? (hideLabel && label ? label : undefined)}
        aria-labelledby={ariaLabelledBy}
        aria-describedby={ariaDescribedBy}
        aria-invalid={error ? true : undefined}
        id={id}
      >
        <SelectValue placeholder={placeholder || label} />
      </SelectTrigger>
      <SelectContent>
        {options.map((option, idx) => {
          if (option.separator) return <SelectSeparator key={`sep-${idx}`} />;
          const content = renderItem ? (
            renderItem(option)
          ) : (
            <div className="flex items-center gap-2">
              {option.icon}
              <span>{option.label}</span>
            </div>
          );
          return (
            <SelectItem
              key={option.value}
              value={option.value}
              disabled={option.disabled}
            >
              {content}
            </SelectItem>
          );
        })}
      </SelectContent>
    </BaseSelect>
  );

  const selectWithError = (
    <div className={cn("relative mb-6", wrapperClassName)}>
      {select}
      <Text
        variant="small-medium"
        as="span"
        tone="danger"
        className={cn(
          "absolute top-full left-0 mt-1 transition-opacity duration-200",
          error ? "opacity-100" : "opacity-0",
        )}
      >
        {error || " "}{" "}
        {/* Always render with space to maintain consistent height calculation */}
      </Text>
    </div>
  );

  return hideLabel ? (
    selectWithError
  ) : (
    <div className="flex flex-col gap-2">
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-1">
          <label htmlFor={id}>
            <Text variant={labelVariant} as="span" className={labelClassName}>
              {label}
            </Text>
          </label>
          {labelTooltip ? (
            <InformationTooltip description={labelTooltip} iconSize={20} />
          ) : null}
        </div>
        {hint}
      </div>
      {selectWithError}
    </div>
  );
}
