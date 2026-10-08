"use client";

import { InformationTooltip } from "@/components/molecules/InformationTooltip/InformationTooltip";
import {
  Select as KobraSelect,
  SelectContent,
  SelectItem,
  SelectSeparator,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { cn } from "@/lib/utils";
import { ReactNode, useId, useState } from "react";
import type { FieldSize } from "../Input/fieldVariants";
import { Text } from "../Text/Text";
import type { Variant } from "../Text/helpers";

// Kobra's trigger sets its height through `data-size`; the house heights win
// by matching that variant.
const triggerSizeClasses: Record<FieldSize, string> = {
  sm: "data-[size=default]:h-8 ps-3 pe-2 text-xs",
  md: "data-[size=default]:h-9 ps-3 pe-2.5 text-sm",
  lg: "data-[size=default]:h-10 ps-4 pe-3 text-sm",
};

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
  /** Links the visible label to the trigger. Generated when omitted. */
  id?: string;
  hideLabel?: boolean;
  error?: string;
  hint?: ReactNode;
  placeholder?: string;
  className?: string;
  disabled?: boolean;
  value?: string;
  onValueChange?: (value: string) => void;
  options: SelectOption[];
  size?: FieldSize;
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
  const generatedId = useId();
  const triggerId = id ?? generatedId;
  const [uncontrolledValue, setUncontrolledValue] = useState<string>();
  const currentValue = value ?? uncontrolledValue ?? "";

  function renderOption(option: SelectOption) {
    if (renderItem) return renderItem(option);
    return (
      <>
        {option.icon}
        <span>{option.label}</span>
      </>
    );
  }

  // Base UI reads the trigger's text from `items`, so the popup never has to
  // mount to show the selected option.
  const items = options
    .filter((option) => !option.separator)
    .map((option) => ({ value: option.value, label: renderOption(option) }));

  function handleValueChange(nextValue: string | null) {
    if (nextValue === null) return;
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
    <KobraSelect
      value={currentValue === "" ? null : currentValue}
      onValueChange={handleValueChange}
      disabled={disabled}
      items={items}
    >
      <SelectTrigger
        className={cn("w-full", triggerSizeClasses[size], className)}
        aria-label={ariaLabel ?? (hideLabel && label ? label : undefined)}
        aria-labelledby={ariaLabelledBy}
        aria-describedby={ariaDescribedBy}
        aria-invalid={error ? true : undefined}
        id={triggerId}
      >
        <SelectValue placeholder={placeholder || label} />
      </SelectTrigger>
      <SelectContent align="start" alignItemWithTrigger={false}>
        {options.map((option, idx) => {
          if (option.separator) return <SelectSeparator key={`sep-${idx}`} />;
          return (
            <SelectItem
              key={option.value}
              value={option.value}
              disabled={option.disabled}
            >
              {renderOption(option)}
            </SelectItem>
          );
        })}
      </SelectContent>
    </KobraSelect>
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
          <label htmlFor={triggerId}>
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
