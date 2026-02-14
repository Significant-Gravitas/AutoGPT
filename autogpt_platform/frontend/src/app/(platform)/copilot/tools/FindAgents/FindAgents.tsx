"use client";

import { ToolUIPart } from "ai";
import { MorphingTextAnimation } from "../../components/MorphingTextAnimation/MorphingTextAnimation";
import {
  ContentBadge,
  ContentCard,
  ContentCardDescription,
  ContentCardHeader,
  ContentCardTitle,
  ContentGrid,
  ContentLink,
} from "../../components/ToolAccordion/AccordionContent";
import { ToolAccordion } from "../../components/ToolAccordion/ToolAccordion";
import {
  AccordionIcon,
  getAgentHref,
  getAnimationText,
  getFindAgentsOutput,
  getSourceLabelFromToolType,
  isAgentsFoundOutput,
  isErrorOutput,
  ToolIcon,
} from "./helpers";

export interface FindAgentsToolPart {
  type: string;
  toolCallId: string;
  state: ToolUIPart["state"];
  input?: unknown;
  output?: unknown;
}

interface Props {
  part: FindAgentsToolPart;
}

export function FindAgentsTool({ part }: Props) {
  const text = getAnimationText(part);
  const output = getFindAgentsOutput(part);
  const isStreaming =
    part.state === "input-streaming" || part.state === "input-available";
  const isError =
    part.state === "output-error" || (!!output && isErrorOutput(output));

  const query =
    typeof part.input === "object" && part.input !== null
      ? String((part.input as { query?: unknown }).query ?? "").trim()
      : "";

  const agentsFoundOutput =
    part.state === "output-available" && output && isAgentsFoundOutput(output)
      ? output
      : null;

  const hasAgents =
    !!agentsFoundOutput &&
    agentsFoundOutput.agents.length > 0 &&
    (typeof agentsFoundOutput.count !== "number" ||
      agentsFoundOutput.count > 0);
  const totalCount = agentsFoundOutput ? agentsFoundOutput.count : 0;
  const { source } = getSourceLabelFromToolType(part.type);
  const scopeText =
    source === "library"
      ? "in your library"
      : source === "marketplace"
        ? "in marketplace"
        : "";
  const accordionDescription = `Found ${totalCount}${scopeText ? ` ${scopeText}` : ""}${
    query ? ` for "${query}"` : ""
  }`;

  return (
    <div className="py-2">
      <div className="flex items-center gap-2 text-sm text-muted-foreground">
        <ToolIcon
          toolType={part.type}
          isStreaming={isStreaming}
          isError={isError}
        />
        <MorphingTextAnimation
          text={text}
          className={isError ? "text-red-500" : undefined}
        />
      </div>

      {hasAgents && agentsFoundOutput && (
        <ToolAccordion
          icon={<AccordionIcon toolType={part.type} />}
          title="Agent results"
          description={accordionDescription}
        >
          <ContentGrid className="sm:grid-cols-2">
            {agentsFoundOutput.agents.map((agent) => {
              const href = getAgentHref(agent);
              const agentSource =
                agent.source === "library"
                  ? "Library"
                  : agent.source === "marketplace"
                    ? "Marketplace"
                    : null;
              return (
                <ContentCard key={agent.id}>
                  <ContentCardHeader
                    action={
                      href ? <ContentLink href={href}>Open</ContentLink> : null
                    }
                  >
                    <div className="flex items-center gap-2">
                      <ContentCardTitle>{agent.name}</ContentCardTitle>
                      {agentSource && (
                        <ContentBadge>{agentSource}</ContentBadge>
                      )}
                    </div>
                    <ContentCardDescription className="mt-1 line-clamp-2">
                      {agent.description}
                    </ContentCardDescription>
                  </ContentCardHeader>
                </ContentCard>
              );
            })}
          </ContentGrid>
        </ToolAccordion>
      )}
    </div>
  );
}
