"use client";

import * as AccordionPrimitive from "@radix-ui/react-accordion";
import { ArrowDown01Icon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";
import * as React from "react";

import { cn } from "@/lib/utils";

const BaseAccordion = AccordionPrimitive.Root;

// How the parts look together. `card`: sections stacked in one bordered card,
// a line between them (a long form split into steps). `panel`: one section
// whose rows sit flush in a bordered panel (a group of saved credentials).
// `plain`: no rule under a section and a shorter header that never
// underlines (a section inside a node or a form).
const VARIANTS = {
  plain: {
    root: "",
    item: "border-none",
    trigger: "py-2 hover:no-underline",
    content: "",
  },
  card: {
    root: "rounded-xl border border-border-base bg-surface shadow-subtle [&>*+*]:border-t [&>*+*]:border-border-base",
    item: "border-0 px-4",
    trigger: "hover:no-underline",
    content: "px-1 pt-0 pb-4",
  },
  panel: {
    root: "rounded-2xl border border-border-base bg-surface",
    item: "border-b-0",
    trigger: "px-3 py-3 pr-5 hover:no-underline [&>svg]:text-foreground-base",
    content: "px-0 pt-0 pb-0",
  },
} as const;

type AccordionVariant = keyof typeof VARIANTS;

const VariantContext = React.createContext<AccordionVariant | undefined>(
  undefined,
);

function useVariantClass(part: keyof (typeof VARIANTS)[AccordionVariant]) {
  const variant = React.useContext(VariantContext);
  return variant && VARIANTS[variant][part];
}

const BaseAccordionItem = React.forwardRef<
  React.ElementRef<typeof AccordionPrimitive.Item>,
  React.ComponentPropsWithoutRef<typeof AccordionPrimitive.Item>
>(({ className, ...props }, ref) => {
  const variantClass = useVariantClass("item");
  return (
    <AccordionPrimitive.Item
      ref={ref}
      className={cn("border-b", variantClass, className)}
      {...props}
    />
  );
});
BaseAccordionItem.displayName = "AccordionItem";

const BaseAccordionTrigger = React.forwardRef<
  React.ElementRef<typeof AccordionPrimitive.Trigger>,
  React.ComponentPropsWithoutRef<typeof AccordionPrimitive.Trigger>
>(({ className, children, ...props }, ref) => {
  const variantClass = useVariantClass("trigger");
  return (
    <AccordionPrimitive.Header className="flex">
      <AccordionPrimitive.Trigger
        ref={ref}
        className={cn(
          "flex flex-1 items-center justify-between py-4 text-left text-body leading-5 font-medium transition-all hover:underline [&[data-state=open]>svg]:rotate-180",
          variantClass,
          className,
        )}
        {...props}
      >
        {children}
        <Icon
          aria-hidden
          icon={ArrowDown01Icon}
          tone="muted"
          motion="transform"
          className="h-4 w-4 shrink-0"
        />
      </AccordionPrimitive.Trigger>
    </AccordionPrimitive.Header>
  );
});
BaseAccordionTrigger.displayName = AccordionPrimitive.Trigger.displayName;

const BaseAccordionContent = React.forwardRef<
  React.ElementRef<typeof AccordionPrimitive.Content>,
  React.ComponentPropsWithoutRef<typeof AccordionPrimitive.Content>
>(({ className, children, ...props }, ref) => {
  const variantClass = useVariantClass("content");
  return (
    <AccordionPrimitive.Content
      ref={ref}
      className="overflow-hidden text-body leading-5 data-[state=closed]:animate-accordion-up data-[state=open]:animate-accordion-down"
      {...props}
    >
      <div className={cn("pt-0 pb-4", variantClass, className)}>{children}</div>
    </AccordionPrimitive.Content>
  );
});
BaseAccordionContent.displayName = AccordionPrimitive.Content.displayName;

const Accordion = React.forwardRef<
  React.ElementRef<typeof BaseAccordion>,
  React.ComponentPropsWithoutRef<typeof BaseAccordion> & {
    variant?: AccordionVariant;
  }
>(({ variant, ...props }, ref) => {
  const rootClass = variant && VARIANTS[variant].root;
  return (
    <VariantContext.Provider value={variant}>
      <BaseAccordion
        ref={ref}
        data-slot="accordion"
        {...props}
        {...(rootClass ? { className: cn(rootClass, props.className) } : {})}
      />
    </VariantContext.Provider>
  );
});
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
