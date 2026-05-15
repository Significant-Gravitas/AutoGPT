"use client";

import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/molecules/Accordion/Accordion";
import { LightbulbIcon } from "@phosphor-icons/react";

interface Props {
  children: React.ReactNode;
}

export function ReasoningCollapse({ children }: Props) {
  return (
    <Accordion type="single" collapsible className="my-1">
      <AccordionItem value="reasoning" className="border-none">
        <AccordionTrigger className="justify-start gap-1.5 py-1 text-xs font-medium text-zinc-500 hover:no-underline">
          <span className="flex items-center gap-1.5">
            <LightbulbIcon size={14} weight="bold" className="shrink-0" />
            Reasoning
          </span>
        </AccordionTrigger>
        <AccordionContent className="pb-1 pl-5 pt-0 text-xs text-zinc-500 [&_pre]:m-0 [&_pre]:whitespace-pre-wrap [&_pre]:bg-transparent [&_pre]:p-0 [&_pre]:text-xs [&_pre]:text-zinc-500">
          {children}
        </AccordionContent>
      </AccordionItem>
    </Accordion>
  );
}
