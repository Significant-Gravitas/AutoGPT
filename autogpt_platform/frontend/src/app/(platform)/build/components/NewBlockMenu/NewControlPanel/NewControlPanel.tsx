import { Separator } from "@/components/ui/separator";
import { cn } from "@/lib/utils";
import React, { useMemo } from "react";
import { BlockMenu } from "../BlockMenu/BlockMenu";
import { useNewControlPanel } from "./useNewControlPanel";
import { NewSaveControl } from "../SaveControl/NewSaveControl";
import { GraphExecutionID } from "@/lib/autogpt-server-api";
import { history } from "@/app/(platform)/build/components/legacy-builder/history";
import { ControlPanelButton } from "../ControlPanelButton";
import { ArrowUUpLeftIcon, ArrowUUpRightIcon } from "@phosphor-icons/react";
import { GraphSearchMenu } from "../GraphMenu/GraphMenu";
import { CustomNode } from "@/app/(platform)/build/components/legacy-builder/CustomNode/CustomNode";
import { Flag, useGetFlag } from "@/services/feature-flags/use-get-flag";

export type Control = {
  icon: React.ReactNode;
  label: string;
  disabled?: boolean;
  onClick: () => void;
};

interface ControlPanelProps {
  className?: string;
  flowExecutionID: GraphExecutionID | undefined;
  visualizeBeads: "no" | "static" | "animate";
  pinSavePopover: boolean;
  pinBlocksPopover: boolean;
  nodes: CustomNode[];
  onNodeSelect: (nodeId: string) => void;
  onNodeHover?: (nodeId: string | null) => void;
}

export const NewControlPanel = ({
  flowExecutionID,
  visualizeBeads,
  pinSavePopover,
  pinBlocksPopover,
  nodes,
  onNodeSelect,
  onNodeHover,
  className,
}: ControlPanelProps) => {
  const isGraphSearchEnabled = useGetFlag(Flag.GRAPH_SEARCH);

  const {
    blockMenuSelected,
    setBlockMenuSelected,
    agentDescription,
    setAgentDescription,
    saveAgent,
    agentName,
    setAgentName,
    savedAgent,
    isSaving,
    isRunning,
    isStopping,
  } = useNewControlPanel({ flowExecutionID, visualizeBeads });

  const controls: Control[] = useMemo(
    () => [
      {
        label: "Undo",
        icon: <ArrowUUpLeftIcon size={20} weight="bold" />,
        onClick: history.undo,
        disabled: !history.canUndo(),
      },
      {
        label: "Redo",
        icon: <ArrowUUpRightIcon size={20} weight="bold" />,
        onClick: history.redo,
        disabled: !history.canRedo(),
      },
    ],
    [],
  );

  return (
    <section
      className={cn(
        "absolute left-4 top-24 z-10 w-[4.25rem] overflow-hidden rounded-[1rem] border-none bg-white p-0 shadow-[0_1px_5px_0_rgba(0,0,0,0.1)]",
        className,
      )}
    >
      <div className="flex flex-col items-center justify-center rounded-[1rem] p-0">
        <BlockMenu
          pinBlocksPopover={pinBlocksPopover}
          blockMenuSelected={blockMenuSelected}
          setBlockMenuSelected={setBlockMenuSelected}
        />
        <Separator className="text-[#E1E1E1]" />
        {isGraphSearchEnabled && (
          <>
            <GraphSearchMenu
              nodes={nodes}
              blockMenuSelected={blockMenuSelected}
              setBlockMenuSelected={setBlockMenuSelected}
              onNodeSelect={onNodeSelect}
              onNodeHover={onNodeHover}
            />
            <Separator className="text-[#E1E1E1]" />
          </>
        )}
        {controls.map((control, index) => (
          <ControlPanelButton
            key={index}
            onClick={() => control.onClick()}
            data-id={`control-button-${index}`}
            data-testid={`blocks-control-${control.label.toLowerCase()}-button`}
            disabled={control.disabled || false}
            className="rounded-none"
          >
            {control.icon}
          </ControlPanelButton>
        ))}
        <Separator className="text-[#E1E1E1]" />
        <NewSaveControl
          agentMeta={savedAgent}
          canSave={!isSaving && !isRunning && !isStopping}
          onSave={saveAgent}
          agentDescription={agentDescription}
          onDescriptionChange={setAgentDescription}
          agentName={agentName}
          onNameChange={setAgentName}
          pinSavePopover={pinSavePopover}
          blockMenuSelected={blockMenuSelected}
          setBlockMenuSelected={setBlockMenuSelected}
        />
      </div>
    </section>
  );
};

export default NewControlPanel;
