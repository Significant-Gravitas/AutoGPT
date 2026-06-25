import { Button } from "@/components/__legacy__/ui/button";
import { Skeleton } from "@/components/__legacy__/ui/skeleton";
import { beautifyString, cn } from "@/lib/utils";
import React, { ButtonHTMLAttributes, useCallback, useState } from "react";
import { highlightText } from "./helpers";
import { PlusIcon } from "@phosphor-icons/react";
import { BlockInfo } from "@/app/api/__generated__/models/blockInfo";
import { useControlPanelStore } from "../../../stores/controlPanelStore";
import { blockDragPreviewStyle } from "./style";
import { useReactFlow } from "@xyflow/react";
import { useNodeStore } from "../../../stores/nodeStore";
import { BlockUIType, SpecialBlockID } from "@/lib/autogpt-server-api";
import {
  MCPToolDialog,
  type MCPToolDialogResult,
} from "@/app/(platform)/build/components/MCPToolDialog";

interface Props extends ButtonHTMLAttributes<HTMLButtonElement> {
  title?: string;
  description?: string;
  highlightedText?: string;
  blockData: BlockInfo;
}

interface BlockComponent extends React.FC<Props> {
  Skeleton: React.FC<{ className?: string }>;
}

export const Block: BlockComponent = ({
  title,
  description,
  highlightedText,
  className,
  blockData,
  ...rest
}) => {
  const setBlockMenuOpen = useControlPanelStore(
    (state) => state.setBlockMenuOpen,
  );
  const { setViewport } = useReactFlow();
  const { addBlock } = useNodeStore();
  const [mcpDialogOpen, setMcpDialogOpen] = useState(false);

  const isMCPBlock = blockData.uiType === BlockUIType.MCP_TOOL;

  const addBlockAndCenter = useCallback(
    (block: BlockInfo, hardcodedValues?: Record<string, any>) => {
      const customNode = addBlock(block, hardcodedValues);
      setTimeout(() => {
        setViewport(
          {
            x: -customNode.position.x * 0.8 + window.innerWidth / 2,
            y: -customNode.position.y * 0.8 + (window.innerHeight - 400) / 2,
            zoom: 0.8,
          },
          { duration: 500 },
        );
      }, 50);
      return customNode;
    },
    [addBlock, setViewport],
  );

  const updateNodeData = useNodeStore((state) => state.updateNodeData);

  const handleMCPToolConfirm = useCallback(
    (result: MCPToolDialogResult) => {
      // Derive a display label: prefer server name, fall back to URL hostname.
      let serverLabel = result.serverName;
      if (!serverLabel) {
        try {
          serverLabel = new URL(result.serverUrl).hostname;
        } catch {
          serverLabel = "MCP";
        }
      }

      const customNode = addBlockAndCenter(blockData, {
        server_url: result.serverUrl,
        server_name: serverLabel,
        selected_tool: result.selectedTool,
        tool_input_schema: result.toolInputSchema,
        available_tools: result.availableTools,
        credentials: result.credentials ?? undefined,
      });
      if (customNode) {
        const title = result.selectedTool
          ? `${serverLabel}: ${beautifyString(result.selectedTool)}`
          : undefined;
        updateNodeData(customNode.id, {
          metadata: {
            ...customNode.data.metadata,
            credentials_optional: true,
            ...(title && { customized_name: title }),
          },
        });
      }
      setMcpDialogOpen(false);
    },
    [addBlockAndCenter, blockData, updateNodeData],
  );

  const handleClick = () => {
    if (isMCPBlock) {
      setMcpDialogOpen(true);
      return;
    }
    const customNode = addBlockAndCenter(blockData);
    // Set customized_name for agent blocks so the agent's name persists
    if (customNode && blockData.id === SpecialBlockID.AGENT) {
      updateNodeData(customNode.id, {
        metadata: {
          ...customNode.data.metadata,
          customized_name: blockData.name,
        },
      });
    }
  };

  const handleDragStart = (e: React.DragEvent<HTMLButtonElement>) => {
    if (isMCPBlock) return;
    e.dataTransfer.effectAllowed = "copy";
    e.dataTransfer.setData("application/reactflow", JSON.stringify(blockData));

    setBlockMenuOpen(false);

    // preview when user drags it
    const dragPreview = document.createElement("div");
    dragPreview.style.cssText = blockDragPreviewStyle;
    dragPreview.textContent = beautifyString(title || "").replace(
      / Block$/,
      "",
    );

    document.body.appendChild(dragPreview);
    e.dataTransfer.setDragImage(dragPreview, 0, 0);

    setTimeout(() => document.body.removeChild(dragPreview), 0);
  };

  // Generate a data-id from the block id (e.g., "AgentInputBlock" -> "block-card-AgentInputBlock")
  const blockDataId = blockData.id
    ? `block-card-${blockData.id.replace(/[^a-zA-Z0-9]/g, "")}`
    : undefined;

  return (
    <>
      <Button
        draggable={!isMCPBlock}
        data-id={blockDataId}
        className={cn(
          "group flex h-16 w-full min-w-[7.5rem] items-center justify-start space-x-3 whitespace-normal rounded-[0.75rem] bg-zinc-50 px-[0.875rem] py-[0.625rem] text-start shadow-none",
          "hover:cursor-default hover:bg-zinc-100 focus:ring-0 active:bg-zinc-100 active:ring-1 active:ring-zinc-300 disabled:cursor-not-allowed",
          isMCPBlock && "hover:cursor-pointer",
          className,
        )}
        onDragStart={handleDragStart}
        onClick={handleClick}
        {...rest}
      >
        <div className="flex flex-1 flex-col items-start gap-0.5">
          {title && (
            <span
              className={cn(
                "line-clamp-1 font-sans text-sm font-medium leading-[1.375rem] text-zinc-800 group-disabled:text-zinc-400",
              )}
            >
              {highlightText(
                beautifyString(title).replace(/ Block$/, ""),
                highlightedText,
              )}
            </span>
          )}
          {description && (
            <span
              className={cn(
                "line-clamp-1 font-sans text-xs font-normal leading-5 text-zinc-500 group-disabled:text-zinc-400",
              )}
            >
              {highlightText(description, highlightedText)}
            </span>
          )}
        </div>
        <div
          className={cn(
            "flex h-7 w-7 items-center justify-center rounded-[0.5rem] bg-zinc-700 group-disabled:bg-zinc-400",
          )}
        >
          <PlusIcon className="h-5 w-5 text-zinc-50" />
        </div>
      </Button>
      {isMCPBlock && (
        <MCPToolDialog
          open={mcpDialogOpen}
          onClose={() => setMcpDialogOpen(false)}
          onConfirm={handleMCPToolConfirm}
        />
      )}
    </>
  );
};

const BlockSkeleton = () => {
  return (
    <Skeleton className="flex h-16 w-full min-w-[7.5rem] animate-pulse items-center justify-start space-x-3 rounded-[0.75rem] bg-zinc-100 px-[0.875rem] py-[0.625rem]">
      <div className="flex flex-1 flex-col items-start gap-0.5">
        <Skeleton className="h-[1.375rem] w-24 rounded bg-zinc-200" />
        <Skeleton className="h-5 w-32 rounded bg-zinc-200" />
      </div>
      <Skeleton className="h-7 w-7 rounded-[0.5rem] bg-zinc-200" />
    </Skeleton>
  );
};

Block.Skeleton = BlockSkeleton;
