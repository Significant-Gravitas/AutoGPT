"use client";

import { Text } from "@/components/atoms/Text/Text";
import { Checkbox as KobraCheckbox } from "@/components/ui/checkbox";
import { cn } from "@/lib/utils";
import { forwardRef, ReactNode, useId } from "react";
import {
  checkboxSizeClasses,
  getDescribedBy,
  INDETERMINATE_CLASSES,
} from "./helpers";

type KobraCheckboxProps = React.ComponentProps<typeof KobraCheckbox>;

interface Props extends Omit<
  KobraCheckboxProps,
  "checked" | "onCheckedChange" | "children" | "className" | "shape" | "render"
> {
  label?: ReactNode;
  description?: ReactNode;
  error?: string;
  size?: "sm" | "md";
  className?: string;
  /** `"indeterminate"` shows a partial selection; the next toggle reports a boolean. */
  checked?: boolean | "indeterminate";
  onCheckedChange?: (checked: boolean) => void;
}

export const Checkbox = forwardRef<HTMLButtonElement, Props>(function Checkbox(
  {
    id,
    label,
    description,
    error,
    size = "sm",
    className,
    checked,
    onCheckedChange,
    "aria-describedby": ariaDescribedBy,
    ...props
  },
  ref,
) {
  const generatedId = useId();
  const checkboxId = id ?? generatedId;
  const descriptionId = description ? `${checkboxId}-description` : undefined;
  const errorId = error ? `${checkboxId}-error` : undefined;
  const indeterminate = checked === "indeterminate";

  const box = (
    <KobraCheckbox
      ref={ref as React.Ref<HTMLElement>}
      id={checkboxId}
      checked={indeterminate ? false : checked}
      indeterminate={indeterminate}
      onCheckedChange={
        onCheckedChange ? (next) => onCheckedChange(next) : undefined
      }
      aria-invalid={error ? true : undefined}
      aria-describedby={getDescribedBy(ariaDescribedBy, descriptionId, errorId)}
      className={cn(
        checkboxSizeClasses[size],
        INDETERMINATE_CLASSES,
        label ? (size === "sm" ? "mt-0.5" : "mt-px") : undefined,
        className,
      )}
      {...props}
    />
  );

  if (!label && !description && !error) return box;

  return (
    <div className="flex items-start gap-2">
      {box}
      <div className="flex min-w-0 flex-col gap-0.5">
        {label ? (
          <label
            htmlFor={checkboxId}
            className={cn(
              "cursor-pointer",
              props.disabled && "cursor-not-allowed opacity-50",
            )}
          >
            <Text variant="body-medium" as="span" tone="primary">
              {label}
            </Text>
          </label>
        ) : null}
        {description ? (
          <Text variant="small" tone="secondary" id={descriptionId}>
            {description}
          </Text>
        ) : null}
        {error ? (
          <Text variant="small" tone="danger" id={errorId}>
            {error}
          </Text>
        ) : null}
      </div>
    </div>
  );
});
