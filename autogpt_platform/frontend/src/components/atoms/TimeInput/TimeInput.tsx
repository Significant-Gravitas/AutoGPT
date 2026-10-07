import React, { ReactNode } from "react";
import { cn } from "@/lib/utils";
import { Text } from "../Text/Text";
import { fieldVariants, type FieldSize } from "../Input/fieldVariants";

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
  size?: FieldSize;
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
  size = "lg",
  wrapperClassName,
  "aria-label": ariaLabel,
  "aria-labelledby": ariaLabelledBy,
  "aria-describedby": ariaDescribedBy,
}) => {
  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    onChange?.(e.target.value);
  };

  const input = (
    <div className={cn("relative", wrapperClassName)}>
      <input
        type="time"
        value={value}
        onChange={handleChange}
        className={cn(
          fieldVariants({ size, invalid: Boolean(error) }),
          className,
        )}
        disabled={disabled}
        placeholder={placeholder || label}
        aria-label={ariaLabel ?? (hideLabel && label ? label : undefined)}
        aria-labelledby={ariaLabelledBy}
        aria-describedby={ariaDescribedBy}
        aria-invalid={error ? true : undefined}
        id={id}
      />
    </div>
  );

  const inputWithError = (
    <div className={cn("relative mb-6", wrapperClassName)}>
      {input}
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

  return hideLabel || !label ? (
    inputWithError
  ) : (
    <label htmlFor={id} className="flex flex-col gap-2">
      <div className="flex items-center justify-between">
        <Text variant="body-medium" as="span">
          {label}
        </Text>
        {hint ? (
          <Text variant="small" as="span" tone="muted">
            {hint}
          </Text>
        ) : null}
      </div>
      {inputWithError}
    </label>
  );
};
