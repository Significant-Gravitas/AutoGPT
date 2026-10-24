import React, { ReactNode } from "react";
import { cn } from "@/lib/utils";
import { Text } from "../Text/Text";

interface TimeInputProps {
  value?: string;
  onChange?: (value: string) => void;
  className?: string;
  disabled?: boolean;
  placeholder?: string;
  label?: string;
  id?: string;
  hideLabel?: boolean;
  error?: string;
  hint?: ReactNode;
  size?: "small" | "medium";
  wrapperClassName?: string;
  "aria-label"?: string;
  "aria-labelledby"?: string;
  "aria-describedby"?: string;
}

export const TimeInput: React.FC<TimeInputProps> = ({
  value = "",
  onChange,
  className,
  disabled = false,
  placeholder = "HH:MM",
  label,
  id,
  hideLabel = false,
  error,
  hint,
  size = "medium",
  wrapperClassName,
  "aria-label": ariaLabel,
  "aria-labelledby": ariaLabelledBy,
  "aria-describedby": ariaDescribedBy,
}) => {
  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    onChange?.(e.target.value);
  };

  const baseStyles = cn(
    // Base styles
    "rounded-3xl border border-border-base bg-surface px-4 shadow-none",
    "font-normal text-foreground-base",
    "placeholder:font-normal placeholder:text-placeholder",
    // Focus and hover states
    "focus:border-foreground-subtle focus:shadow-none focus:outline-none focus:ring-1 focus:ring-foreground-subtle focus:ring-offset-0",
    className,
  );

  const errorStyles =
    error && "!border !border-error focus:border-error focus:ring-error";

  const input = (
    <div
      data-slot="time-input-wrapper"
      className={cn("relative", wrapperClassName)}
    >
      <input
        data-slot="time-input-control"
        type="time"
        value={value}
        onChange={handleChange}
        className={cn(
          baseStyles,
          errorStyles,
          // Size variants
          size === "small" && [
            "h-9", // 36px
            "py-2",
            "text-body", // 14px font, 22px line height
            "placeholder:text-body",
          ],
          size === "medium" && [
            "h-[2.875rem]", // 46px (current default)
            "py-2.5",
          ],
        )}
        disabled={disabled}
        placeholder={placeholder || label}
        aria-label={ariaLabel ?? (hideLabel && label ? label : undefined)}
        aria-labelledby={ariaLabelledBy}
        aria-describedby={ariaDescribedBy}
        id={id}
      />
    </div>
  );

  const inputWithError = (
    <div
      data-slot="time-input-field"
      className={cn("relative mb-6", wrapperClassName)}
    >
      {input}
      <Text
        data-slot="time-input-error"
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

  return hideLabel || !label ? (
    inputWithError
  ) : (
    <label data-slot="time-input" htmlFor={id} className="flex flex-col gap-2">
      <div
        data-slot="time-input-header"
        className="flex items-center justify-between"
      >
        <Text data-slot="time-input-label" variant="body-medium" as="span">
          {label}
        </Text>
        {hint ? (
          <Text
            data-slot="time-input-hint"
            variant="small"
            as="span"
            tone="subtle"
          >
            {hint}
          </Text>
        ) : null}
      </div>
      {inputWithError}
    </label>
  );
};
