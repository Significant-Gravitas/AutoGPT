import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { beautifyString, cn } from "@/lib/utils";
import { PlusSignIcon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";
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
        "group flex h-16 w-full min-w-[7.5rem] items-center justify-start gap-3 whitespace-normal rounded-xl bg-surface-muted px-3.5 py-2.5 text-start shadow-none",
        "hover:cursor-default hover:bg-surface-hover focus:ring-0 active:bg-surface-hover active:ring-1 active:ring-border-strong disabled:cursor-not-allowed",
        className,
      )}
      onDragStart={handleDragStart}
      onClick={handleClick}
      {...rest}
    >
      <div className="relative h-[2.625rem] w-[2.625rem] rounded-lg bg-surface">
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
          "flex h-7 w-7 items-center justify-center rounded-lg bg-foreground-body text-action-foreground group-disabled:bg-foreground-subtle",
        )}
      >
        <Icon aria-hidden icon={PlusSignIcon} className="h-5 w-5" />
      </div>
    </Button>
  );
};

const IntegrationBlockSkeleton = ({ className }: { className?: string }) => {
  return (
    <Skeleton
      className={cn(
        "flex h-16 w-full min-w-[7.5rem] animate-pulse items-center justify-start gap-3 rounded-xl bg-surface-hover px-3.5 py-2.5",
        className,
      )}
    >
      <Skeleton className="h-[2.625rem] w-[2.625rem] rounded-lg bg-surface-active" />
      <div className="flex flex-1 flex-col items-start gap-0.5">
        <Skeleton className="h-[1.375rem] w-24 rounded bg-surface-active" />
        <Skeleton className="h-5 w-32 rounded bg-surface-active" />
      </div>
      <Skeleton className="h-7 w-7 rounded-lg bg-surface-active" />
    </Skeleton>
  );
};

IntegrationBlock.Skeleton = IntegrationBlockSkeleton;
