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
  size?: "small" | "medium";
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
  size = "medium",
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
    // Base styles matching Input
    "rounded-xl border border-zinc-200 bg-white px-4 shadow-none",
    "font-normal text-black w-full",
    "placeholder:font-normal placeholder:text-zinc-500",
    // Focus and hover states
    "focus:border-zinc-400 focus:shadow-none focus:outline-none focus:ring-1 focus:ring-zinc-400 focus:ring-offset-0",
    // Size variants
    size === "small" && [
      "h-[2.25rem]",
      "py-2",
      "text-sm leading-[22px]",
      "placeholder:text-sm placeholder:leading-[22px]",
    ],
    size === "medium" && ["h-[2.875rem]", "py-2.5", "text-sm"],
    // Error state
    error && "border-red-500 focus:border-red-500 focus:ring-red-500",
    // Placeholder styling for SelectValue when data-placeholder is present
    "[&[data-placeholder]>span]:text-zinc-400 [&[data-placeholder]>span]:font-normal",
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
        className={cn(
          "absolute left-0 top-full mt-1 !text-red-500 transition-opacity duration-200",
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
            <Text
              variant={labelVariant}
              as="span"
              className={cn("text-black", labelClassName)}
            >
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
