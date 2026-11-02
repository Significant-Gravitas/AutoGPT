import { Text } from "@/components/atoms/Text/Text";
import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { beautifyString, cn } from "@/lib/utils";
import React, { ButtonHTMLAttributes, memo, useState } from "react";
import { highlightText } from "./helpers";
import { BlockInfo } from "@/app/api/__generated__/models/blockInfo";
import { useControlPanelStore } from "../../../stores/controlPanelStore";
import {
  blockDragPreviewClassName,
  blockDragPreviewStyle,
  blockMenuButtonBase,
} from "./style";
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

function BlockCard({
  title,
  description,
  highlightedText,
  className,
  blockData,
  ...rest
}: Props) {
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
    dragPreview.className = blockDragPreviewClassName;
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
      <button
        type="button"
        draggable={!isMCPBlock}
        data-id={blockDataId}
        className={cn(
          blockMenuButtonBase,
          "group flex h-16 w-full min-w-[7.5rem] items-center justify-start space-x-3 rounded-xl bg-surface-muted px-3.5 py-2.5 text-start whitespace-normal",
          "hover:cursor-default hover:bg-surface-hover active:bg-surface-hover active:ring-1 active:ring-border-strong disabled:cursor-not-allowed",
          isMCPBlock && "hover:cursor-pointer",
          className,
        )}
        onDragStart={handleDragStart}
        onClick={handleClick}
        {...rest}
      >
        <div className="flex flex-1 flex-col items-start gap-0.5">
          {title && (
            <Text
              variant="body-medium"
              as="span"
              tone="emphasis"
              className={cn(
                "line-clamp-1 group-disabled:text-foreground-subtle",
              )}
            >
              {highlightText(
                beautifyString(title).replace(/ Block$/, ""),
                highlightedText,
              )}
            </Text>
          )}
          {description && (
            <Text
              variant="small-loose"
              as="span"
              tone="muted"
              className={cn(
                "line-clamp-1 group-disabled:text-foreground-subtle",
              )}
            >
              {highlightText(description, highlightedText, "small-bold-loose")}
            </Text>
          )}
        </div>
        <div
          className={cn(
            "flex h-7 w-7 items-center justify-center rounded-lg bg-foreground-body group-disabled:bg-foreground-subtle",
          )}
        >
          <Icon
            icon={PlusSignIcon}
            className="h-5 w-5 text-action-foreground"
          />
        </div>
      </button>
      {isMCPBlock && (
        <MCPToolDialog
          open={mcpDialogOpen}
          onClose={() => setMcpDialogOpen(false)}
          onConfirm={handleMCPToolConfirm}
        />
      )}
    </>
  );
}

const BlockSkeleton = () => {
  return (
    <Skeleton className="flex h-16 w-full min-w-[7.5rem] animate-pulse items-center justify-start space-x-3 rounded-xl bg-surface-hover px-3.5 py-2.5">
      <div className="flex flex-1 flex-col items-start gap-0.5">
        <Skeleton className="h-[1.375rem] w-24 rounded bg-surface-active" />
        <Skeleton className="h-5 w-32 rounded bg-surface-active" />
      </div>
      <Skeleton className="h-7 w-7 rounded-lg bg-surface-active" />
    </Skeleton>
  );
};

// Memoised: a list re-renders for its own reasons (a new page arriving, a
// filter count updating) without the rows' props changing.
export const Block = Object.assign(memo(BlockCard), {
  Skeleton: BlockSkeleton,
});
