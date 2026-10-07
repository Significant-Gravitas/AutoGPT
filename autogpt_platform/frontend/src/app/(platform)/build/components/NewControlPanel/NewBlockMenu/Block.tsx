import { Button } from "@/components/atoms/Button/Button";
import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { beautifyString, cn } from "@/lib/utils";
import React, { ButtonHTMLAttributes, useState } from "react";
import { highlightText } from "./helpers";
import { BlockInfo } from "@/app/api/__generated__/models/blockInfo";
import { useControlPanelStore } from "../../../stores/controlPanelStore";
import { blockDragPreviewStyle } from "./style";
import { useNodeStore } from "../../../stores/nodeStore";
import { useAddBlockToBuilder } from "./hooks/useAddBlockToBuilder";
import { BlockUIType, SpecialBlockID } from "@/lib/autogpt-server-api";
import {
  MCPToolDialog,
  type MCPToolDialogResult,
} from "@/app/(platform)/build/components/MCPToolDialog";
import { PlusSignIcon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";

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
  const { addBlockWithPlacement } = useAddBlockToBuilder();
  const [mcpDialogOpen, setMcpDialogOpen] = useState(false);

  const isMCPBlock = blockData.uiType === BlockUIType.MCP_TOOL;

  const updateNodeData = useNodeStore((state) => state.updateNodeData);

  function handleMCPToolConfirm(result: MCPToolDialogResult) {
    let serverLabel = result.serverName;
    if (!serverLabel) {
      try {
        serverLabel = new URL(result.serverUrl).hostname;
      } catch {
        serverLabel = "MCP";
      }
    }

    const customNode = addBlockWithPlacement(blockData, {
      server_url: result.serverUrl,
      server_name: serverLabel,
      selected_tool: result.selectedTool,
      tool_input_schema: result.toolInputSchema,
      available_tools: result.availableTools,
      credentials: result.credentials ?? undefined,
    });
    if (customNode) {
      const displayName = result.selectedTool
        ? `${serverLabel}: ${beautifyString(result.selectedTool)}`
        : undefined;
      updateNodeData(customNode.id, {
        metadata: {
          ...customNode.data.metadata,
          credentials_optional: true,
          ...(displayName && { customized_name: displayName }),
        },
      });
    }
    setMcpDialogOpen(false);
  }

  function handleClick() {
    if (isMCPBlock) {
      setMcpDialogOpen(true);
      return;
    }
    const customNode = addBlockWithPlacement(blockData);
    if (customNode && blockData.id === SpecialBlockID.AGENT) {
      updateNodeData(customNode.id, {
        metadata: {
          ...customNode.data.metadata,
          customized_name: blockData.name,
        },
      });
    }
  }

  function handleDragStart(e: React.DragEvent<HTMLButtonElement>) {
    if (isMCPBlock) return;
    e.dataTransfer.effectAllowed = "copy";
    e.dataTransfer.setData("application/reactflow", JSON.stringify(blockData));

    setBlockMenuOpen(false);

    const dragPreview = document.createElement("div");
    dragPreview.style.cssText = blockDragPreviewStyle;
    dragPreview.textContent = beautifyString(title || "").replace(
      / Block$/,
      "",
    );

    document.body.appendChild(dragPreview);
    e.dataTransfer.setDragImage(dragPreview, 0, 0);

    setTimeout(() => document.body.removeChild(dragPreview), 0);
  }

  const blockDataId = blockData.id
    ? `block-card-${blockData.id.replace(/[^a-zA-Z0-9]/g, "")}`
    : undefined;

  return (
    <>
      <Button
        variant="ghost"
        draggable={!isMCPBlock}
        data-id={blockDataId}
        className={cn(
          "group flex h-16 w-full min-w-30 items-center justify-start gap-3 rounded-xl bg-zinc-50 px-3.5 py-2.5 text-start whitespace-normal shadow-none",
          "hover:cursor-default hover:bg-zinc-100 focus:ring-0 active:bg-zinc-100 active:ring-1 active:ring-zinc-300 disabled:cursor-not-allowed disabled:opacity-50",
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
                "line-clamp-1 font-sans text-sm leading-5.5 font-medium text-zinc-800 group-disabled:text-zinc-400",
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
                "line-clamp-1 font-sans text-xs leading-5 font-normal text-zinc-500 group-disabled:text-zinc-400",
              )}
            >
              {highlightText(description, highlightedText)}
            </span>
          )}
        </div>
        <div
          className={cn(
            "flex h-7 w-7 items-center justify-center rounded-lg bg-zinc-700 group-disabled:bg-zinc-400",
          )}
        >
          <Icon icon={PlusSignIcon} className="h-5 w-5 text-zinc-50" />
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
    <Skeleton className="flex h-16 w-full min-w-30 animate-pulse items-center justify-start space-x-3 rounded-xl bg-zinc-100 px-3.5 py-2.5">
      <div className="flex flex-1 flex-col items-start gap-0.5">
        <Skeleton className="h-5.5 w-24 rounded-sm bg-zinc-200" />
        <Skeleton className="h-5 w-32 rounded-sm bg-zinc-200" />
      </div>
      <Skeleton className="h-7 w-7 rounded-lg bg-zinc-200" />
    </Skeleton>
  );
};

Block.Skeleton = BlockSkeleton;
