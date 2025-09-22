"use client";

import * as React from "react";
import {
  Select as BaseSelect,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
  SelectSeparator,
} from "@/components/__legacy__/ui/select";
import { cn } from "@/lib/utils";
import { ReactNode } from "react";
import { Text } from "../Text/Text";

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
  renderItem?: (option: SelectOption) => React.ReactNode;
  wrapperClassName?: string;
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
  renderItem,
  wrapperClassName,
}: SelectFieldProps) {
  const triggerStyles = cn(
    // Base styles matching Input
    "rounded-3xl border border-zinc-200 bg-white px-4 shadow-none",
    "font-normal text-black w-full",
    "placeholder:font-normal !placeholder:text-zinc-400",
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
    error &&
      "border-1.5 border-red-500 focus:border-red-500 focus:ring-red-500",
    // Placeholder styling for SelectValue when data-placeholder is present
    "[&[data-placeholder]>span]:text-zinc-400 [&[data-placeholder]>span]:font-normal",
    className,
  );

  const select = (
    <BaseSelect value={value} onValueChange={onValueChange} disabled={disabled}>
      <SelectTrigger
        className={triggerStyles}
        {...(hideLabel ? { "aria-label": label } : {})}
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
    <label htmlFor={id} className="flex flex-col gap-2">
      <div className="flex items-center justify-between">
        <Text variant="body-medium" as="span" className="text-black">
          {label}
        </Text>
        {hint}
      </div>
      {selectWithError}
    </label>
  );
}
