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
  getAgentHref,
  getAnimationText,
  getFindAgentsOutput,
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

  const agentsFoundOutput =
    part.state === "output-available" && output && isAgentsFoundOutput(output)
      ? output
      : null;

  const hasAgents =
    !!agentsFoundOutput &&
    agentsFoundOutput.agents.length > 0 &&
    (typeof agentsFoundOutput.count !== "number" ||
      agentsFoundOutput.count > 0);

  // With results, the accordion header IS the tool row — a single compact
  // line (icon + summary + caret), matching the bash-style tools.
  if (hasAgents && agentsFoundOutput) {
    return (
      <div className="py-1">
        <ToolAccordion
          variant="compact"
          icon={<ToolIcon toolType={part.type} />}
          title={text}
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
                    <div className="flex min-w-0 items-center gap-2">
                      <ContentCardTitle className="min-w-0 flex-1">
                        {agent.name}
                      </ContentCardTitle>
                      {agentSource && (
                        <ContentBadge>{agentSource}</ContentBadge>
                      )}
                    </div>
                    <ContentCardDescription className="mt-1 line-clamp-2 break-words">
                      {agent.description}
                    </ContentCardDescription>
                  </ContentCardHeader>
                </ContentCard>
              );
            })}
          </ContentGrid>
        </ToolAccordion>
      </div>
    );
  }

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
    </div>
  );
}
