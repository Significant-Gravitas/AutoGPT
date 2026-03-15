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
}) => {
  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    onChange?.(e.target.value);
  };

  const baseStyles = cn(
    // Base styles
    "rounded-3xl border border-zinc-200 bg-white px-4 shadow-none",
    "font-normal text-black",
    "placeholder:font-normal placeholder:text-zinc-400",
    // Focus and hover states
    "focus:border-zinc-400 focus:shadow-none focus:outline-none focus:ring-1 focus:ring-zinc-400 focus:ring-offset-0",
    className,
  );

  const errorStyles =
    error && "!border !border-red-500 focus:border-red-500 focus:ring-red-500";

  const input = (
    <div className={cn("relative", wrapperClassName)}>
      <input
        type="time"
        value={value}
        onChange={handleChange}
        className={cn(
          baseStyles,
          errorStyles,
          // Size variants
          size === "small" && [
            "h-[2.25rem]", // 36px
            "py-2",
            "text-sm leading-[22px]", // 14px font, 22px line height
            "placeholder:text-sm placeholder:leading-[22px]",
          ],
          size === "medium" && [
            "h-[2.875rem]", // 46px (current default)
            "py-2.5",
          ],
        )}
        disabled={disabled}
        placeholder={placeholder || label}
        {...(hideLabel ? { "aria-label": label } : {})}
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

  return hideLabel || !label ? (
    inputWithError
  ) : (
    <label htmlFor={id} className="flex flex-col gap-2">
      <div className="flex items-center justify-between">
        <Text variant="body-medium" as="span" className="text-black">
          {label}
        </Text>
        {hint ? (
          <Text variant="small" as="span" className="!text-zinc-400">
            {hint}
          </Text>
        ) : null}
      </div>
      {inputWithError}
    </label>
  );
};
