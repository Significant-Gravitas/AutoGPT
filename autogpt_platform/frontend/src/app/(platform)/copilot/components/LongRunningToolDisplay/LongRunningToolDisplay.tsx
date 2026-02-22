import { PlusCircleIcon } from "@phosphor-icons/react";
import {
  ContentGrid,
  ContentHint,
} from "../../components/ToolAccordion/AccordionContent";
import { ToolAccordion } from "../../components/ToolAccordion/ToolAccordion";
import { MiniGame } from "../../tools/CreateAgent/components/MiniGame/MiniGame";

interface Props {
  /** Whether the tool is currently streaming/executing */
  isStreaming: boolean;
  /** Custom title for the accordion (optional) */
  title?: string;
}

/**
 * Displays a mini-game while a long-running tool executes.
 * Automatically shown for tools marked as is_long_running=True in the backend.
 */
export function LongRunningToolDisplay({ isStreaming, title }: Props) {
  if (!isStreaming) return null;

  const displayTitle =
    title || "This may take a few minutes. Play while you wait.";

  return (
    <ToolAccordion
      icon={<PlusCircleIcon size={32} weight="light" />}
      title={displayTitle}
      defaultExpanded={true}
    >
      <ContentGrid>
        <MiniGame />
        <ContentHint>
          This could take a few minutes — play while you wait!
        </ContentHint>
      </ContentGrid>
    </ToolAccordion>
  );
}
