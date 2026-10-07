"use client";

import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";
import { forwardRef, ReactNode, useId } from "react";
import { getDescribedBy, textareaVariants } from "./helpers";
import { useTextarea } from "./useTextarea";

interface Props extends Omit<
  React.TextareaHTMLAttributes<HTMLTextAreaElement>,
  "size"
> {
  label: string;
  hideLabel?: boolean;
  hint?: ReactNode;
  error?: string;
  size?: "sm" | "md";
  /** Shows a "length / maxLength" counter under the field. */
  showCount?: boolean;
  wrapperClassName?: string;
}

export const Textarea = forwardRef<HTMLTextAreaElement, Props>(
  function Textarea(
    {
      id,
      label,
      hideLabel = false,
      hint,
      error,
      size = "md",
      rows = 3,
      maxLength,
      showCount = maxLength !== undefined,
      className,
      wrapperClassName,
      value,
      defaultValue,
      onChange,
      onKeyDown,
      "aria-describedby": ariaDescribedBy,
      ...props
    },
    ref,
  ) {
    const generatedId = useId();
    const textareaId = id ?? generatedId;
    const hintId = hint ? `${textareaId}-hint` : undefined;
    const errorId = error ? `${textareaId}-error` : undefined;
    const { length, handleChange, handleKeyDown } = useTextarea({
      value,
      defaultValue,
      onChange,
      onKeyDown,
    });
    const showCounter = showCount && maxLength !== undefined;

    return (
      <div className={cn("flex w-full flex-col gap-2", wrapperClassName)}>
        <div
          className={cn(
            "flex items-center justify-between gap-2",
            hideLabel && !hint && "sr-only",
          )}
        >
          <label htmlFor={textareaId} className={cn(hideLabel && "sr-only")}>
            <Text variant="large-medium" as="span">
              {label}
            </Text>
          </label>
          {hint ? (
            <Text variant="small" as="span" tone="secondary" id={hintId}>
              {hint}
            </Text>
          ) : null}
        </div>
        <textarea
          ref={ref}
          id={textareaId}
          rows={rows}
          maxLength={maxLength}
          value={value}
          defaultValue={defaultValue}
          onChange={handleChange}
          onKeyDown={handleKeyDown}
          aria-invalid={error ? true : undefined}
          aria-describedby={getDescribedBy(ariaDescribedBy, hintId, errorId)}
          className={cn(
            textareaVariants({ size, invalid: Boolean(error) }),
            className,
          )}
          {...props}
        />
        {error || showCounter ? (
          <div className="flex items-start justify-between gap-2">
            {error ? (
              <Text variant="small-medium" as="span" tone="danger" id={errorId}>
                {error}
              </Text>
            ) : (
              <span />
            )}
            {showCounter ? (
              <Text
                variant="small"
                as="span"
                tone={length >= maxLength ? "danger" : "secondary"}
                className="shrink-0 tabular-nums"
              >
                {length}/{maxLength}
              </Text>
            ) : null}
          </div>
        ) : null}
      </div>
    );
  },
);
