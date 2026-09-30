"use client";

import { HorizontalScroll } from "@/app/(platform)/build/components/NewControlPanel/NewBlockMenu/HorizontalScroll";
import { MorphingTextAnimation } from "../../components/MorphingTextAnimation/MorphingTextAnimation";
import {
  ContentCard,
  ContentCardDescription,
  ContentCardTitle,
} from "../../components/ToolAccordion/AccordionContent";
import { ToolAccordion } from "../../components/ToolAccordion/ToolAccordion";
import { AccordionIcon, ToolIcon } from "../FindBlocks/helpers";
import {
  type CapabilityListing,
  type FindCapabilityToolPart,
  getAnimationText,
  kindLabel,
  parseOutput,
  queryOf,
} from "./helpers";

interface Props {
  part: FindCapabilityToolPart;
}

function CapabilityCard({ item }: { item: CapabilityListing }) {
  const connected = item.connected;
  return (
    <ContentCard className="w-52 shrink-0">
      <div className="flex items-start justify-between gap-2">
        <ContentCardTitle className="truncate">{item.name}</ContentCardTitle>
        {connected !== undefined && connected !== null && (
          <span
            className={
              "shrink-0 rounded-full px-2 py-0.5 text-[11px] " +
              (connected
                ? "bg-green-100 text-green-800"
                : "bg-zinc-100 text-zinc-500")
            }
          >
            {connected ? "connected" : "sign in"}
          </span>
        )}
      </div>
      <ContentCardDescription className="mt-1 line-clamp-2">
        {item.purpose}
      </ContentCardDescription>
      <p className="mt-1 text-[11px] text-zinc-500">{kindLabel(item)}</p>
    </ContentCard>
  );
}

export function FindCapabilitiesTool({ part }: Props) {
  const text = getAnimationText(part);
  const isStreaming =
    part.state === "input-streaming" || part.state === "input-available";
  const isError = part.state === "output-error";
  const parsed =
    part.state === "output-available" ? parseOutput(part.output) : null;
  const items = parsed
    ? [...parsed.capabilities, ...(parsed.fallback ?? [])]
    : [];
  const query = queryOf(part);
  const description = parsed
    ? `${parsed.count} result${parsed.count === 1 ? "" : "s"}${query ? ` for "${query}"` : ""}`
    : undefined;

  return (
    <div className="py-2">
      <div className="flex items-center gap-2 text-sm text-muted-foreground">
        <ToolIcon isStreaming={isStreaming} isError={isError} />
        <MorphingTextAnimation
          text={text}
          className={isError ? "text-red-500" : undefined}
        />
      </div>

      {items.length > 0 && (
        <ToolAccordion
          icon={<AccordionIcon />}
          title="Results"
          description={description}
        >
          <HorizontalScroll dependencyList={[items.length]}>
            {items.map((item) => (
              <CapabilityCard key={item.id} item={item} />
            ))}
          </HorizontalScroll>
        </ToolAccordion>
      )}
    </div>
  );
}
