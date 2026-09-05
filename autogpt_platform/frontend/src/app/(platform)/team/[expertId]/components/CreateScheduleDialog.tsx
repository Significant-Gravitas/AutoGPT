"use client";

import { useGetV2GetLibraryAgent } from "@/app/api/__generated__/endpoints/library/library";
import { ExpertWorkflowRef } from "@/app/api/__generated__/models/expertWorkflowRef";
import { okData } from "@/app/api/helpers";
import { RunAgentModal } from "@/app/(platform)/library/agents/[id]/components/NewAgentLibraryView/components/modals/RunAgentModal/RunAgentModal";
import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { ArrowRight01Icon, Calendar03Icon } from "@hugeicons/core-free-icons";
import { useCreateScheduleDialog } from "./useCreateScheduleDialog";
import { useFitListToDialog } from "./useFitListToDialog";

interface Props {
  expertId: string;
  workflows: ExpertWorkflowRef[];
  open: boolean;
  onClose: () => void;
}

export function CreateScheduleDialog({
  expertId,
  workflows,
  open,
  onClose,
}: Props) {
  const { handleScheduleCreated } = useCreateScheduleDialog(expertId, onClose);
  const schedulable = workflows.filter((workflow) => workflow.library_agent_id);
  const { attachList } = useFitListToDialog<HTMLUListElement>();

  return (
    <Dialog
      controlled={{
        isOpen: open,
        set: (next) => {
          if (!next) onClose();
        },
      }}
      styling={{ maxWidth: "28rem", maxHeight: "60vh" }}
      title="Create schedule"
    >
      <Dialog.Content>
        <div className="flex flex-col">
          {schedulable.length === 0 ? (
            <Text variant="body" className="text-zinc-600">
              Install a workflow first, then schedule it here.
            </Text>
          ) : (
            <ul
              ref={attachList}
              className="flex flex-col gap-2 overflow-y-auto pr-1"
              aria-label="Schedulable workflows"
            >
              {schedulable.map((workflow) => (
                <li key={workflow.id}>
                  <ScheduleWorkflowRow
                    workflow={workflow}
                    onScheduleCreated={handleScheduleCreated}
                  />
                </li>
              ))}
            </ul>
          )}
          <div className="flex justify-end pt-1">
            <Button variant="secondary" size="small" onClick={onClose}>
              Cancel
            </Button>
          </div>
        </div>
      </Dialog.Content>
    </Dialog>
  );
}

interface RowProps {
  workflow: ExpertWorkflowRef;
  onScheduleCreated: () => void;
}

function ScheduleWorkflowRow({ workflow, onScheduleCreated }: RowProps) {
  const { data: agent } = useGetV2GetLibraryAgent(
    workflow.library_agent_id ?? "",
    { query: { select: okData, enabled: Boolean(workflow.library_agent_id) } },
  );
  const name = workflow.name ?? "Unnamed workflow";
  const row = (
    <button
      type="button"
      disabled={!agent}
      className="flex w-full items-center gap-3 rounded-2xl border border-zinc-200 px-4 py-3 text-left transition-colors hover:bg-zinc-50 disabled:opacity-60"
    >
      <Icon
        icon={Calendar03Icon}
        size={18}
        className="shrink-0 text-zinc-500"
      />
      <span className="min-w-0 flex-1">
        <span className="block truncate text-sm font-medium text-zinc-900">
          {name}
        </span>
        {workflow.description ? (
          <span className="block truncate text-xs text-zinc-500">
            {workflow.description}
          </span>
        ) : null}
      </span>
      <Icon
        icon={ArrowRight01Icon}
        size={16}
        className="shrink-0 text-zinc-400"
      />
    </button>
  );

  if (!agent) return row;
  return (
    <RunAgentModal
      agent={agent}
      triggerSlot={row}
      onScheduleCreated={onScheduleCreated}
    />
  );
}
