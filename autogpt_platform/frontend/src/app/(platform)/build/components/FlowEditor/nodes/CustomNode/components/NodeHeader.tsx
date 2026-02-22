import { useNodeStore } from "@/app/(platform)/build/stores/nodeStore";
import { Text } from "@/components/atoms/Text/Text";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/atoms/Tooltip/BaseTooltip";
import { beautifyString, cn } from "@/lib/utils";
import { useEffect, useState } from "react";
import { CustomNodeData } from "../CustomNode";
import { NodeBadges } from "./NodeBadges";
import { NodeContextMenu } from "./NodeContextMenu";
import { NodeCost } from "./NodeCost";

type Props = {
  data: CustomNodeData;
  nodeId: string;
};

export function NodeHeader({ data, nodeId }: Props) {
  const updateNodeData = useNodeStore((state) => state.updateNodeData);

  // For Agent Executor blocks, show agent name + version if available
  function getTitle(): string {
    if (data.metadata?.customized_name) {
      return data.metadata.customized_name;
    }

    const agentName = data.hardcodedValues?.agent_name;
    const agentVersion = data.hardcodedValues?.graph_version;

    if (agentName && agentVersion != null) {
      return `${agentName} v${agentVersion}`;
    } else if (agentName) {
      return agentName;
    }

    return data.title;
  }

  function getBeautifiedTitle(): string {
    const rawTitle = getTitle();

    // Extract version suffix (e.g., "v1", "v2.0", "v3.1.4") to preserve casing
    const versionMatch = rawTitle.match(/\s+(v\d+(?:\.\d+)*)$/i);

    if (versionMatch) {
      const nameOnly = rawTitle.slice(0, -versionMatch[0].length);
      const versionSuffix = versionMatch[1]; // Preserve original case
      return beautifyString(nameOnly).replace("Block", "").trim() + " " + versionSuffix;
    }

    return beautifyString(rawTitle).replace("Block", "").trim();
  }

  const title = getBeautifiedTitle();

  const [isEditingTitle, setIsEditingTitle] = useState(false);
  const [editedTitle, setEditedTitle] = useState(title);

  // Sync editedTitle when title changes (e.g., agent updated)
  // Only sync when NOT editing to preserve user's in-progress edits
  useEffect(() => {
    if (!isEditingTitle) {
      setEditedTitle(title);
    }
  }, [title, isEditingTitle]);

  const handleTitleEdit = () => {
    // Only persist if the title actually changed to avoid freezing agent-derived titles
    if (editedTitle.trim() !== title.trim()) {
      updateNodeData(nodeId, {
        metadata: { ...data.metadata, customized_name: editedTitle },
      });
    }
    setIsEditingTitle(false);
  };

  const handleTitleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter") handleTitleEdit();
    if (e.key === "Escape") {
      setEditedTitle(title);
      setIsEditingTitle(false);
    }
  };

  return (
    <div className="flex h-auto flex-col gap-1 rounded-xlarge border-b border-zinc-200 bg-gradient-to-r from-slate-50/80 to-white/90 px-4 py-4 pt-3">
      {/* Title row with context menu */}
      <div className="flex items-start justify-between gap-2">
        <div className="flex min-w-0 flex-1 items-center gap-2">
          <div
            onDoubleClick={() => setIsEditingTitle(true)}
            className="flex w-fit min-w-0 flex-1 items-center hover:cursor-pointer"
          >
            {isEditingTitle ? (
              <input
                id="node-title-input"
                value={editedTitle}
                onChange={(e) => setEditedTitle(e.target.value)}
                autoFocus
                className={cn(
                  "m-0 h-fit w-full border-none bg-transparent p-0 focus:outline-none focus:ring-0",
                  "font-sans text-[1rem] font-semibold leading-[1.5rem] text-zinc-800",
                )}
                onBlur={handleTitleEdit}
                onKeyDown={handleTitleKeyDown}
              />
            ) : (
              <TooltipProvider>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <div>
                      <Text
                        variant="large-semibold"
                        className="line-clamp-1 hover:cursor-text"
                      >
                        {title}
                      </Text>
                    </div>
                  </TooltipTrigger>
                  <TooltipContent>
                    <p>{title}</p>
                  </TooltipContent>
                </Tooltip>
              </TooltipProvider>
            )}
          </div>

          <div className="flex items-center gap-2">
            <Text
              variant="small"
              className="shrink-0 !font-medium !text-slate-500"
            >
              #{nodeId.split("-")[0]}
            </Text>
            <NodeContextMenu
              subGraphID={data.hardcodedValues?.graph_id}
              nodeId={nodeId}
            />
          </div>
        </div>
      </div>

      {/* Metadata row */}
      <div className="flex flex-wrap items-center gap-2">
        <NodeCost blockCosts={data.costs} nodeId={nodeId} />
        <NodeBadges categories={data.categories} />
      </div>
    </div>
  );
}
