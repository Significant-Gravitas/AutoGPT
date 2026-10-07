"use client";

import * as React from "react";
import { Calendar03Icon } from "@hugeicons/core-free-icons";
import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";
import { fieldVariants, type FieldSize } from "../Input/fieldVariants";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/__legacy__/ui/popover";
import { Calendar } from "@/components/__legacy__/ui/calendar";

function toLocalISODateString(d: Date) {
  const year = d.getFullYear();
  const month = String(d.getMonth() + 1).padStart(2, "0");
  const day = String(d.getDate()).padStart(2, "0");
  return `${year}-${month}-${day}`;
}

function parseISODateString(s?: string): Date | undefined {
  if (!s) return undefined;
  // Expecting "YYYY-MM-DD"
  const m = /^(\d{4})-(\d{2})-(\d{2})$/.exec(s);
  if (!m) return undefined;
  const [_, y, mo, d] = m;
  const date = new Date(Number(y), Number(mo) - 1, Number(d));
  return isNaN(date.getTime()) ? undefined : date;
}

export interface DateInputProps {
  value?: string;
  onChange?: (value?: string) => void;
  disabled?: boolean;
  readonly?: boolean;
  placeholder?: string;
  autoFocus?: boolean;
  className?: string;
  label?: string;
  hideLabel?: boolean;
  error?: string;
  id?: string;
  size?: FieldSize;
  "aria-label"?: string;
  "aria-labelledby"?: string;
  "aria-describedby"?: string;
}

export const DateInput = ({
  value,
  onChange,
  disabled,
  readonly,
  placeholder,
  autoFocus,
  className,
  label,
  hideLabel = false,
  error,
  id,
  size = "lg",
  "aria-label": ariaLabel,
  "aria-labelledby": ariaLabelledBy,
  "aria-describedby": ariaDescribedBy,
}: DateInputProps) => {
  const selected = React.useMemo(() => parseISODateString(value), [value]);
  const [open, setOpen] = React.useState(false);

  const setDate = (d?: Date) => {
    onChange?.(d ? toLocalISODateString(d) : undefined);
    setOpen(false);
  };

  const buttonText =
    selected?.toLocaleDateString(undefined, {
      year: "numeric",
      month: "short",
      day: "numeric",
    }) ||
    placeholder ||
    "Pick a date";

  const isDisabled = disabled || readonly;

  const triggerStyles = cn(
    fieldVariants({ size, invalid: Boolean(error) }),
    "min-w-0 justify-start gap-2 text-left",
    !selected && "text-muted-foreground",
    className,
  );

  return (
    <div className="flex flex-col gap-1">
      {label && !hideLabel && (
        <label htmlFor={id}>
          <Text variant="body-medium" as="span">
            {label}
          </Text>
        </label>
      )}
      <Popover open={open} onOpenChange={setOpen}>
        <PopoverTrigger asChild>
          <Button
            type="button"
            variant="ghost"
            className={triggerStyles}
            disabled={isDisabled}
            autoFocus={autoFocus}
            id={id}
            aria-label={ariaLabel ?? (hideLabel && label ? label : undefined)}
            aria-labelledby={ariaLabelledBy}
            aria-describedby={ariaDescribedBy}
            aria-invalid={error ? true : undefined}
          >
            <Icon
              icon={Calendar03Icon}
              size={size === "sm" ? 14 : 16}
              aria-hidden
            />
            {buttonText}
          </Button>
        </PopoverTrigger>
        <PopoverContent className="w-auto p-0" sideOffset={6}>
          <Calendar
            mode="single"
            selected={selected}
            defaultMonth={selected}
            onSelect={setDate}
            showOutsideDays
            // Prevent selection when disabled/readonly
            modifiersClassNames={{
              disabled: "pointer-events-none opacity-50",
            }}
          />
        </PopoverContent>
      </Popover>
      {error && (
        <Text variant="small-medium" as="span" tone="danger">
          {error}
        </Text>
      )}
    </div>
  );
};
