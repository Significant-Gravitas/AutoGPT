"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { BookOpenIcon, PencilSimpleIcon } from "@phosphor-icons/react";
import Image from "next/image";
import sparklesImg from "../MiniGame/assets/sparkles.png";
import { useCopilotChatActions } from "../CopilotChatActionsProvider/useCopilotChatActions";

interface Props {
  agentName: string;
  message: string;
  libraryAgentLink: string;
  agentPageLink: string;
}

export function AgentSavedCard({
  agentName,
  message,
  libraryAgentLink,
  agentPageLink,
}: Props) {
  // On the in-builder chat panel the user is already looking at the agent
  // and the panel auto-switches the URL to the new version — the two
  // "Open in …" CTAs here would open the same agent in a second tab,
  // which is confusing. Keep the status line; hide the buttons.
  const { chatSurface } = useCopilotChatActions();
  const hideNavButtons = chatSurface === "builder";

  return (
    <div className="rounded-xl border border-border/60 bg-card p-4 shadow-sm">
      <div className="flex items-baseline gap-2">
        <Image
          src={sparklesImg}
          alt="sparkles"
          width={24}
          height={24}
          className="relative top-1"
        />
        <Text variant="body-medium" className="mb-2 text-[16px] text-black">
          Agent <span className="text-violet-600">{agentName}</span> {message}
        </Text>
      </div>
      {!hideNavButtons && (
        <div className="mt-3 flex flex-wrap gap-2">
          <Button
            size="small"
            as="NextLink"
            href={libraryAgentLink}
            target="_blank"
            rel="noopener noreferrer"
          >
            <BookOpenIcon size={14} weight="regular" />
            Open in library
          </Button>
          <Button
            as="NextLink"
            variant="secondary"
            size="small"
            href={agentPageLink}
            target="_blank"
            rel="noopener noreferrer"
          >
            <PencilSimpleIcon size={14} weight="regular" />
            Open in builder
          </Button>
        </div>
      )}
    </div>
  );
}
