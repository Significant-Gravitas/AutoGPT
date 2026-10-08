"use client";

import {
  Accordion as KobraAccordion,
  AccordionContent as KobraAccordionContent,
  AccordionItem as KobraAccordionItem,
  AccordionTrigger as KobraAccordionTrigger,
} from "@/components/ui/accordion";
import { cn } from "@/lib/utils";
import type { ComponentProps } from "react";

type WithClassName<T> = Omit<T, "className"> & { className?: string };

type RootProps = WithClassName<
  Omit<
    ComponentProps<typeof KobraAccordion>,
    "value" | "defaultValue" | "onValueChange" | "multiple"
  >
>;

interface SingleProps extends RootProps {
  type: "single";
  /** Lets the open item be closed again. */
  collapsible?: boolean;
  value?: string;
  defaultValue?: string;
  onValueChange?: (value: string) => void;
}

interface MultipleProps extends RootProps {
  type: "multiple";
  value?: string[];
  defaultValue?: string[];
  onValueChange?: (value: string[]) => void;
}

export type AccordionProps = SingleProps | MultipleProps;

export function Accordion(props: AccordionProps) {
  if (props.type === "multiple") {
    const { type: _type, onValueChange, ...rest } = props;
    return (
      <KobraAccordion
        multiple
        onValueChange={(next) => onValueChange?.(next as string[])}
        {...rest}
      />
    );
  }

  const {
    type: _type,
    collapsible = false,
    value,
    defaultValue,
    onValueChange,
    ...rest
  } = props;

  return (
    <KobraAccordion
      multiple={false}
      value={value === undefined ? undefined : value ? [value] : []}
      defaultValue={defaultValue ? [defaultValue] : undefined}
      onValueChange={(next, details) => {
        const nextValue: string = next[0] ?? "";
        if (!collapsible && nextValue === "") {
          details.cancel();
          return;
        }
        onValueChange?.(nextValue);
      }}
      {...rest}
    />
  );
}

export function AccordionItem({
  className,
  ...props
}: WithClassName<ComponentProps<typeof KobraAccordionItem>>) {
  return (
    <KobraAccordionItem
      className={cn("border-t-0 border-b border-border", className)}
      {...props}
    />
  );
}

interface AccordionTriggerProps extends WithClassName<
  ComponentProps<typeof KobraAccordionTrigger>
> {
  /** Hides the expand mark for triggers that draw their own state indicator. */
  showChevron?: boolean;
}

export function AccordionTrigger({
  className,
  showChevron = true,
  ...props
}: AccordionTriggerProps) {
  return (
    <KobraAccordionTrigger
      className={cn(
        "py-4 text-sm font-medium text-foreground",
        !showChevron && "[&>svg:last-child]:hidden",
        className,
      )}
      {...props}
    />
  );
}

export function AccordionContent({
  className,
  ...props
}: WithClassName<ComponentProps<typeof KobraAccordionContent>>) {
  return (
    <KobraAccordionContent
      className={cn("pb-4 text-sm text-foreground", className)}
      {...props}
    />
  );
}
