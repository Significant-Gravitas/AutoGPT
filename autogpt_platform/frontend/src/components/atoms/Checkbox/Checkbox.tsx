"use client";

import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";
import { MinusSignIcon, Tick02Icon } from "@hugeicons/core-free-icons";
import * as CheckboxPrimitive from "@radix-ui/react-checkbox";
import { forwardRef, ReactNode, useId } from "react";
import { checkboxVariants, getDescribedBy, indicatorSize } from "./helpers";

interface Props extends Omit<
  React.ComponentPropsWithoutRef<typeof CheckboxPrimitive.Root>,
  "children"
> {
  label?: ReactNode;
  description?: ReactNode;
  error?: string;
  size?: "sm" | "md";
}

export const Checkbox = forwardRef<
  React.ElementRef<typeof CheckboxPrimitive.Root>,
  Props
>(function Checkbox(
  {
    id,
    label,
    description,
    error,
    size = "sm",
    className,
    "aria-describedby": ariaDescribedBy,
    ...props
  },
  ref,
) {
  const generatedId = useId();
  const checkboxId = id ?? generatedId;
  const descriptionId = description ? `${checkboxId}-description` : undefined;
  const errorId = error ? `${checkboxId}-error` : undefined;

  const box = (
    <CheckboxPrimitive.Root
      ref={ref}
      id={checkboxId}
      aria-invalid={error ? true : undefined}
      aria-describedby={getDescribedBy(ariaDescribedBy, descriptionId, errorId)}
      className={cn(
        checkboxVariants({ size, invalid: Boolean(error) }),
        label ? (size === "sm" ? "mt-0.5" : "mt-px") : undefined,
        className,
      )}
      {...props}
    >
      <CheckboxPrimitive.Indicator className="flex items-center justify-center text-current">
        <Icon
          icon={Tick02Icon}
          size={indicatorSize[size]}
          strokeWidth={2.5}
          className="group-data-[state=indeterminate]:hidden"
          aria-hidden
        />
        <Icon
          icon={MinusSignIcon}
          size={indicatorSize[size]}
          strokeWidth={2.5}
          className="hidden group-data-[state=indeterminate]:block"
          aria-hidden
        />
      </CheckboxPrimitive.Indicator>
    </CheckboxPrimitive.Root>
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
