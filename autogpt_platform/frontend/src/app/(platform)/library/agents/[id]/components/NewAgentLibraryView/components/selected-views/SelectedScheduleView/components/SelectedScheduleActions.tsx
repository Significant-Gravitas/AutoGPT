import { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { Button } from "@/components/atoms/Button/Button";
import { EyeIcon } from "@phosphor-icons/react";
import { AgentActionsDropdown } from "../../AgentActionsDropdown";
import { useScheduleDetailHeader } from "../../RunDetailHeader/useScheduleDetailHeader";
import { SelectedActionsWrap } from "../../SelectedActionsWrap";

type Props = {
  agent: LibraryAgent;
  scheduleId: string;
  onDeleted?: () => void;
};

export function SelectedScheduleActions({ agent, scheduleId }: Props) {
  const { openInBuilderHref } = useScheduleDetailHeader(
    agent.graph_id,
    scheduleId,
    agent.graph_version,
  );

  return (
    <>
      <SelectedActionsWrap>
        {openInBuilderHref && (
          <Button
            variant="icon"
            size="icon"
            as="NextLink"
            href={openInBuilderHref}
            target="_blank"
            aria-label="View scheduled task details"
          >
            <EyeIcon weight="bold" size={18} className="text-zinc-700" />
          </Button>
        )}
        <AgentActionsDropdown agent={agent} scheduleId={scheduleId} />
      </SelectedActionsWrap>
    </>
  );
}
