import { useNodeStore } from "@/app/(platform)/build/stores/nodeStore";
import { Text } from "@/components/atoms/Text/Text";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/atoms/Tooltip/BaseTooltip";
import { beautifyString, cn } from "@/lib/utils";
import { useEffect, useMemo, useRef, useState } from "react";
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
      return data.metadata.customized_name as string;
    }

    const agentName = data.hardcodedValues?.agent_name as string | undefined;
    const agentVersion = data.hardcodedValues?.graph_version as
      | number
      | undefined;

    if (agentName && agentVersion != null) {
      return `${agentName} v${agentVersion}`;
    } else if (agentName) {
      return agentName;
    }

    return data.title;
  }

  function getBeautifiedTitle(): string {
    const rawTitle = getTitle();

    // Don't beautify user-customized names or agent-derived names
    if (data.metadata?.customized_name || data.hardcodedValues?.agent_name) {
      return rawTitle;
    }

    // Only for block type names (from data.title): beautify and strip "Block" suffix
    const beautified = beautifyString(rawTitle);

    // Strip "Block" suffix only if it's at the end (e.g., "AITextGeneratorBlock" -> "AI Text Generator")
    // Don't strip "Block" from middle of names (e.g., "Blockchain Analyzer" stays "Blockchain Analyzer")
    if (beautified.endsWith(" Block")) {
      return beautified.slice(0, -6).trim();
    }

    return beautified.trim();
  }

  // Memoize title to prevent unnecessary recalculations
  const title = useMemo(
    () => getBeautifiedTitle(),
    [
      data.metadata?.customized_name,
      data.hardcodedValues?.agent_name,
      data.hardcodedValues?.graph_version,
      data.title,
    ],
  );

  const [isEditingTitle, setIsEditingTitle] = useState(false);
  const [editedTitle, setEditedTitle] = useState(title);

  // Track the last synced title to detect external changes
  const lastSyncedTitle = useRef(title);

  // Sync editedTitle when title changes externally (e.g., agent updated)
  // Only sync when NOT editing to preserve user's in-progress edits
  useEffect(() => {
    if (!isEditingTitle && title !== lastSyncedTitle.current) {
      setEditedTitle(title);
      lastSyncedTitle.current = title;
    }
  }, [title, isEditingTitle]);

  const handleTitleEdit = () => {
    const trimmedEditedTitle = editedTitle.trim();
    const trimmedOriginalTitle = title.trim();

    // Only persist if the title actually changed to avoid freezing agent-derived titles
    // Allow saving empty string if user explicitly wants to clear it
    if (trimmedEditedTitle !== trimmedOriginalTitle) {
      updateNodeData(nodeId, {
        metadata: {
          ...data.metadata,
          customized_name: trimmedEditedTitle || undefined, // Clear customized_name if empty
        },
      });
      lastSyncedTitle.current = trimmedEditedTitle || title;
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
