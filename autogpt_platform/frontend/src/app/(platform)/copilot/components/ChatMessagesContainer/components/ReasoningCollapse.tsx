"use client";

import { Button } from "@/components/atoms/Button/Button";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
} from "@/components/molecules/Accordion/Accordion";
import { cn } from "@/lib/utils";
import { LightbulbIcon } from "@phosphor-icons/react";
import * as AccordionPrimitive from "@radix-ui/react-accordion";
import { useState } from "react";

interface Props {
  children: React.ReactNode;
  /**
   * True while the reasoning is still being produced by the model. Drives
   * the pulse animation on the trigger — once the part is done streaming
   * the indicator stops pulsing so it doesn't look like the chat is still
   * "thinking" when it isn't.
   */
  isActive?: boolean;
}

export function ReasoningCollapse({ children, isActive = false }: Props) {
  const [value, setValue] = useState("");

  return (
    <Accordion
      type="single"
      collapsible
      className="my-1"
      value={value}
      onValueChange={(newValue) => setValue(newValue === value ? "" : newValue)}
    >
      <AccordionItem value="reasoning" className="border-none">
        <AccordionPrimitive.Header className="flex">
          <AccordionPrimitive.Trigger
            className={cn(
              "flex items-center gap-1.5 py-1 text-xs font-medium text-zinc-500 transition-colors hover:text-zinc-700 focus-visible:outline-none",
              isActive && "animate-pulse",
            )}
          >
            <LightbulbIcon size={14} weight="bold" className="shrink-0" />
            Reasoning
          </AccordionPrimitive.Trigger>
        </AccordionPrimitive.Header>
        <AccordionContent className="pb-1 pt-0 text-xs text-zinc-500 [&_pre]:m-0 [&_pre]:whitespace-pre-wrap [&_pre]:bg-transparent [&_pre]:p-0 [&_pre]:text-xs [&_pre]:text-zinc-500">
          {children}
          <Button
            variant="secondary"
            size="small"
            className="mt-2"
            onClick={() => setValue("")}
          >
            Collapse
          </Button>
        </AccordionContent>
      </AccordionItem>
    </Accordion>
  );
}
