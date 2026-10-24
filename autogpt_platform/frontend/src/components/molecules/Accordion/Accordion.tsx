"use client";

import * as React from "react";

import {
  Accordion as BaseAccordion,
  AccordionContent as BaseAccordionContent,
  AccordionItem as BaseAccordionItem,
  AccordionTrigger as BaseAccordionTrigger,
} from "@/components/ui/accordion";

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

const AccordionTrigger = React.forwardRef<
  React.ElementRef<typeof BaseAccordionTrigger>,
  React.ComponentPropsWithoutRef<typeof BaseAccordionTrigger>
>((props, ref) => (
  <BaseAccordionTrigger ref={ref} data-slot="accordion-trigger" {...props} />
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
