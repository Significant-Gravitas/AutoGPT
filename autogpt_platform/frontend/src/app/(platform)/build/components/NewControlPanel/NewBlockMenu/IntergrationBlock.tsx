import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { beautifyString, cn } from "@/lib/utils";
import { Plus } from "lucide-react";
import Image from "next/image";
import React, { ButtonHTMLAttributes } from "react";
import { highlightText } from "./helpers";
import { Button } from "@/components/atoms/Button/Button";
import { useControlPanelStore } from "../../../stores/controlPanelStore";
import { useAddBlockToBuilder } from "./hooks/useAddBlockToBuilder";
import { BlockInfo } from "@/app/api/__generated__/models/blockInfo";
import { blockDragPreviewClassName, blockDragPreviewStyle } from "./style";
import { Text } from "@/components/atoms/Text/Text";

interface Props extends ButtonHTMLAttributes<HTMLButtonElement> {
  title?: string;
  description?: string;
  icon_url?: string;
  highlightedText?: string;
  blockData: BlockInfo;
}

interface IntegrationBlockComponent extends React.FC<Props> {
  Skeleton: React.FC<{ className?: string }>;
}

export const IntegrationBlock: IntegrationBlockComponent = ({
  title,
  icon_url,
  description,
  className,
  highlightedText,
  blockData,
  ...rest
}) => {
  const setBlockMenuOpen = useControlPanelStore(
    (state) => state.setBlockMenuOpen,
  );
  const { addBlockWithPlacement } = useAddBlockToBuilder();

  function handleClick() {
    addBlockWithPlacement(blockData);
  }

  function handleDragStart(e: React.DragEvent<HTMLButtonElement>) {
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

  return (
    <Button
      draggable={true}
      variant={"ghost"}
      className={cn(
        "group flex h-16 w-full min-w-[7.5rem] items-center justify-start gap-3 whitespace-normal rounded-[0.75rem] bg-surface-muted px-[0.875rem] py-[0.625rem] text-start shadow-none",
        // TODO(tokens): no fitting role for active:bg-zinc-100 #EFEFF0; #EFEFF0 is surface-hover, a hover fill, and this is the pressed state (QUALITY 41); no fitting role for active:ring-zinc-300 #C5C5C9; #C5C5C9 is border-strong (a border colour) and foreground-disabled (a text colour), and this is a ring (QUALITY 41(b))
        "hover:cursor-default hover:bg-surface-hover focus:ring-0 active:bg-zinc-100 active:ring-1 active:ring-zinc-300 disabled:cursor-not-allowed",
        className,
      )}
      onDragStart={handleDragStart}
      onClick={handleClick}
      {...rest}
    >
      <div className="relative h-[2.625rem] w-[2.625rem] rounded-[0.5rem] bg-surface">
        {icon_url && (
          <Image
            src={icon_url}
            alt="integration-icon"
            fill
            sizes="2.25rem"
            className="w-full object-contain group-disabled:opacity-50"
          />
        )}
      </div>
      <div className="flex flex-1 flex-col items-start gap-0.5">
        {title && (
          <Text
            variant="body-medium"
            as="span"
            tone="emphasis"
            className={cn("line-clamp-1 group-disabled:text-foreground-subtle")}
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
            className={cn("line-clamp-1 group-disabled:text-foreground-subtle")}
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
        <Plus
          // TODO(tokens): no fitting role for text-zinc-50 #F9F9FA; #F9F9FA is surface-muted (a surface fill)
          className="h-5 w-5 text-zinc-50"
          strokeWidth={2}
        />
      </div>
    </Button>
  );
};

const IntegrationBlockSkeleton = ({ className }: { className?: string }) => {
  return (
    <Skeleton
      className={cn(
        // TODO(tokens): no fitting role for bg-zinc-100 #EFEFF0; #EFEFF0 is surface-hover, a hover fill, and this fill is not a hover state (QUALITY 41)
        "flex h-16 w-full min-w-[7.5rem] animate-pulse items-center justify-start gap-3 rounded-[0.75rem] bg-zinc-100 px-[0.875rem] py-[0.625rem]",
        className,
      )}
    >
      <Skeleton
        // TODO(tokens): no fitting role for bg-zinc-200 #DADADC; #DADADC is surface-active, an active-state fill, and border-base (a border colour), and this fill is static (QUALITY 41)
        className="h-[2.625rem] w-[2.625rem] rounded-[0.5rem] bg-zinc-200"
      />
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

IntegrationBlock.Skeleton = IntegrationBlockSkeleton;
