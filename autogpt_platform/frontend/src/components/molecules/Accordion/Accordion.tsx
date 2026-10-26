"use client";

import * as AccordionPrimitive from "@radix-ui/react-accordion";
import { ChevronDown } from "lucide-react";
import * as React from "react";

import { cn } from "@/lib/utils";

const BaseAccordion = AccordionPrimitive.Root;

const BaseAccordionItem = React.forwardRef<
  React.ElementRef<typeof AccordionPrimitive.Item>,
  React.ComponentPropsWithoutRef<typeof AccordionPrimitive.Item>
>(({ className, ...props }, ref) => (
  <AccordionPrimitive.Item
    ref={ref}
    className={cn("border-b", className)}
    {...props}
  />
));
BaseAccordionItem.displayName = "AccordionItem";

const BaseAccordionTrigger = React.forwardRef<
  React.ElementRef<typeof AccordionPrimitive.Trigger>,
  React.ComponentPropsWithoutRef<typeof AccordionPrimitive.Trigger>
>(({ className, children, ...props }, ref) => (
  <AccordionPrimitive.Header className="flex">
    <AccordionPrimitive.Trigger
      ref={ref}
      className={cn(
        "flex flex-1 items-center justify-between py-4 text-left text-body font-medium leading-5 transition-all hover:underline [&[data-state=open]>svg]:rotate-180",
        className,
      )}
      {...props}
    >
      {children}
      <ChevronDown className="h-4 w-4 shrink-0 text-neutral-500 transition-transform duration-200 dark:text-neutral-400" />
    </AccordionPrimitive.Trigger>
  </AccordionPrimitive.Header>
));
BaseAccordionTrigger.displayName = AccordionPrimitive.Trigger.displayName;

const BaseAccordionContent = React.forwardRef<
  React.ElementRef<typeof AccordionPrimitive.Content>,
  React.ComponentPropsWithoutRef<typeof AccordionPrimitive.Content>
>(({ className, children, ...props }, ref) => (
  <AccordionPrimitive.Content
    ref={ref}
    className="overflow-hidden text-body leading-5 data-[state=closed]:animate-accordion-up data-[state=open]:animate-accordion-down"
    {...props}
  >
    <div className={cn("pb-4 pt-0", className)}>{children}</div>
  </AccordionPrimitive.Content>
));
BaseAccordionContent.displayName = AccordionPrimitive.Content.displayName;

const Accordion = React.forwardRef<
  React.ElementRef<typeof BaseAccordion>,
  React.ComponentPropsWithoutRef<typeof BaseAccordion>
>((props, ref) => <BaseAccordion ref={ref} data-slot="accordion" {...props} />);
Accordion.displayName = "Accordion";

const AccordionItem = React.forwardRef<
  React.ElementRef<typeof BaseAccordionItem>,
  React.ComponentPropsWithoutRef<typeof BaseAccordionItem>
>((props, ref) => (
  <BaseAccordionItem ref={ref} data-slot="accordion-item" {...props} />
));
AccordionItem.displayName = "AccordionItem";

interface TextSizeProps {
  /** `body-md` inside a profile tab, whose copy reads at 15px. */
  textSize?: "body" | "body-md";
}

const AccordionTrigger = React.forwardRef<
  React.ElementRef<typeof BaseAccordionTrigger>,
  React.ComponentPropsWithoutRef<typeof BaseAccordionTrigger> & TextSizeProps
>(({ textSize = "body", className, ...props }, ref) => (
  <BaseAccordionTrigger
    ref={ref}
    data-slot="accordion-trigger"
    className={
      textSize === "body-md" ? cn(className, "text-body-md") : className
    }
    {...props}
  />
));
AccordionTrigger.displayName = "AccordionTrigger";

const AccordionContent = React.forwardRef<
  React.ElementRef<typeof BaseAccordionContent>,
  React.ComponentPropsWithoutRef<typeof BaseAccordionContent>
>((props, ref) => (
  <BaseAccordionContent ref={ref} data-slot="accordion-panel" {...props} />
));
AccordionContent.displayName = "AccordionContent";

export { Accordion, AccordionContent, AccordionItem, AccordionTrigger };
