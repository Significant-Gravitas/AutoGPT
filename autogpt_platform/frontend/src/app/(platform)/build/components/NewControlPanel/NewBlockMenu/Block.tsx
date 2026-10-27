import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { beautifyString, cn } from "@/lib/utils";
import React, { ButtonHTMLAttributes, memo, useState } from "react";
import { highlightText } from "./helpers";
import { BlockInfo } from "@/app/api/__generated__/models/blockInfo";
import { useControlPanelStore } from "../../../stores/controlPanelStore";
import { blockDragPreviewClassName, blockDragPreviewStyle } from "./style";
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
      <Button
        variant="legacy-default"
        draggable={!isMCPBlock}
        data-id={blockDataId}
        className={cn(
          "group flex h-16 w-full min-w-[7.5rem] items-center justify-start space-x-3 whitespace-normal rounded-[0.75rem] bg-surface-muted px-[0.875rem] py-[0.625rem] text-start shadow-none",
          // TODO(tokens): no fitting role for active:bg-zinc-100 #EFEFF0; #EFEFF0 is surface-hover, a hover fill, and this is the pressed state (QUALITY 41); no fitting role for active:ring-zinc-300 #C5C5C9; #C5C5C9 is border-strong (a border colour) and foreground-disabled (a text colour), and this is a ring (QUALITY 41(b))
          "hover:cursor-default hover:bg-surface-hover focus:ring-0 active:bg-zinc-100 active:ring-1 active:ring-zinc-300 disabled:cursor-not-allowed",
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
            // TODO(tokens): no fitting role for bg-zinc-700 #505057; #505057 is foreground-body (a text colour); no fitting role for group-disabled:bg-zinc-400 #ADADB3; #ADADB3 is foreground-subtle (a text colour)
            "flex h-7 w-7 items-center justify-center rounded-[0.5rem] bg-zinc-700 group-disabled:bg-zinc-400",
          )}
        >
          <Icon
            // TODO(tokens): no fitting role for text-zinc-50 #F9F9FA; #F9F9FA is surface-muted (a surface fill)
            icon={PlusSignIcon}
            className="h-5 w-5 text-zinc-50"
          />
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
}

const BlockSkeleton = () => {
  return (
    <Skeleton
      // TODO(tokens): no fitting role for bg-zinc-100 #EFEFF0; #EFEFF0 is surface-hover, a hover fill, and this fill is not a hover state (QUALITY 41)
      className="flex h-16 w-full min-w-[7.5rem] animate-pulse items-center justify-start space-x-3 rounded-[0.75rem] bg-zinc-100 px-[0.875rem] py-[0.625rem]"
    >
      <div className="flex flex-1 flex-col items-start gap-0.5">
        <Skeleton
          // TODO(tokens): no fitting role for bg-zinc-200 #DADADC; #DADADC is surface-active, an active-state fill, and border-base (a border colour), and this fill is static (QUALITY 41)
          className="h-[1.375rem] w-24 rounded bg-zinc-200"
        />
        <Skeleton
          // TODO(tokens): no fitting role for bg-zinc-200 #DADADC; #DADADC is surface-active, an active-state fill, and border-base (a border colour), and this fill is static (QUALITY 41)
          className="h-5 w-32 rounded bg-zinc-200"
        />
      </div>
      <Skeleton
        // TODO(tokens): no fitting role for bg-zinc-200 #DADADC; #DADADC is surface-active, an active-state fill, and border-base (a border colour), and this fill is static (QUALITY 41)
        className="h-7 w-7 rounded-[0.5rem] bg-zinc-200"
      />
    </Skeleton>
  );
};

// Memoised: a list re-renders for its own reasons (a new page arriving, a
// filter count updating) without the rows' props changing.
export const Block = Object.assign(memo(BlockCard), {
  Skeleton: BlockSkeleton,
});
