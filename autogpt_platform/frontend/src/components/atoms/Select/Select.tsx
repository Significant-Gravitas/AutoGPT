"use client";

import {
  Select as BaseSelect,
  SelectContent,
  SelectItem,
  SelectSeparator,
  SelectTrigger,
  SelectValue,
} from "./components/SelectPrimitives";
import { cn } from "@/lib/utils";
import * as React from "react";
import { ReactNode } from "react";
import { Text } from "../Text/Text";
import type { Variant } from "../Text/helpers";
import { InformationTooltip } from "@/components/molecules/InformationTooltip/InformationTooltip";

export interface SelectOption {
  value: string;
  label: string;
  icon?: ReactNode;
  disabled?: boolean;
  separator?: boolean;
  onSelect?: () => void; // optional action handler
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
    "rounded-xl border border-border-base bg-surface px-4 shadow-none",
    "font-normal text-foreground-base w-full",
    "placeholder:font-normal !placeholder:text-placeholder",
    // Focus and hover states
    "focus:border-foreground-subtle focus:shadow-none focus:outline-none focus:ring-1 focus:ring-foreground-subtle focus:ring-offset-0",
    // Size variants
    size === "small" && ["h-9", "py-2", "text-body", "placeholder:text-body"],
    size === "medium" && ["h-[2.875rem]", "py-2.5", "text-body leading-5"],
    // Error state
    error && "border-1.5 border-error focus:border-error focus:ring-error",
    // Placeholder styling for SelectValue when data-placeholder is present
    "[&[data-placeholder]>span]:text-foreground-subtle [&[data-placeholder]>span]:font-normal",
    className,
  );

  const select = (
    <BaseSelect value={value} onValueChange={onValueChange} disabled={disabled}>
      <SelectTrigger
        data-slot="select-trigger"
        className={triggerStyles}
        aria-label={ariaLabel ?? (hideLabel && label ? label : undefined)}
        aria-labelledby={ariaLabelledBy}
        aria-describedby={ariaDescribedBy}
        id={id}
      >
        <SelectValue
          data-slot="select-value"
          placeholder={placeholder || label}
        />
      </SelectTrigger>
      <SelectContent data-slot="select-popup">
        {options.map((option, idx) => {
          if (option.separator)
            return (
              <SelectSeparator
                key={`sep-${idx}`}
                data-slot="select-separator"
              />
            );
          const content = renderItem ? (
            renderItem(option)
          ) : (
            <div
              data-slot="select-item-content"
              className="flex items-center gap-2"
            >
              {option.icon}
              <span data-slot="select-item-label">{option.label}</span>
            </div>
          );
          return (
            <SelectItem
              key={option.value}
              data-slot="select-item"
              value={option.value}
              disabled={option.disabled}
              onMouseDown={(e) => {
                if (option.onSelect) {
                  e.preventDefault();
                  option.onSelect();
                }
              }}
            >
              {content}
            </SelectItem>
          );
        })}
      </SelectContent>
    </BaseSelect>
  );

  const selectWithError = (
    <div
      data-slot="select-field"
      className={cn("relative mb-6", wrapperClassName)}
    >
      {select}
      <Text
        data-slot="select-error"
        variant="small-medium"
        as="span"
        tone="error"
        className={cn(
          "absolute left-0 top-full mt-1 transition-opacity duration-200",
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
    <label data-slot="select" htmlFor={id} className="flex flex-col gap-2">
      <div
        data-slot="select-header"
        className="flex items-center justify-between gap-2"
      >
        <div data-slot="select-label-group" className="flex items-center gap-1">
          <Text
            data-slot="select-label"
            variant={labelVariant}
            as="span"
            className={cn("text-foreground-base", labelClassName)}
          >
            {label}
          </Text>
          {labelTooltip ? (
            <InformationTooltip description={labelTooltip} iconSize={20} />
          ) : null}
        </div>
        {hint}
      </div>
      {selectWithError}
    </label>
  );
}
