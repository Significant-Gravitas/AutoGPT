import { Skeleton } from "@/components/__legacy__/ui/skeleton";
import { beautifyString, cn } from "@/lib/utils";
import { Plus } from "lucide-react";
import Image from "next/image";
import React, { ButtonHTMLAttributes } from "react";
import { highlightText } from "./helpers";
import { Button } from "@/components/atoms/Button/Button";
import { useControlPanelStore } from "../../../stores/controlPanelStore";
import { useReactFlow } from "@xyflow/react";
import { useNodeStore } from "../../../stores/nodeStore";
import { BlockInfo } from "@/app/api/__generated__/models/blockInfo";
import { blockDragPreviewStyle } from "./style";

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
  const { setViewport } = useReactFlow();
  const { addBlock } = useNodeStore();

  const handleClick = () => {
    const customNode = addBlock(blockData);
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
  };

  const handleDragStart = (e: React.DragEvent<HTMLButtonElement>) => {
    e.dataTransfer.effectAllowed = "copy";
    e.dataTransfer.setData("application/reactflow", JSON.stringify(blockData));

    setBlockMenuOpen(false);

    // preview when user drags it
    const dragPreview = document.createElement("div");
    dragPreview.style.cssText = blockDragPreviewStyle;
    dragPreview.textContent = beautifyString(title || "");

    document.body.appendChild(dragPreview);
    e.dataTransfer.setDragImage(dragPreview, 0, 0);

    setTimeout(() => document.body.removeChild(dragPreview), 0);
  };

  return (
    <Button
      draggable={true}
      variant={"ghost"}
      className={cn(
        "group flex h-16 w-full min-w-[7.5rem] items-center justify-start gap-3 whitespace-normal rounded-[0.75rem] bg-zinc-50 px-[0.875rem] py-[0.625rem] text-start shadow-none",
        "hover:cursor-default hover:bg-zinc-100 focus:ring-0 active:bg-zinc-100 active:ring-1 active:ring-zinc-300 disabled:cursor-not-allowed",
        className,
      )}
      onDragStart={handleDragStart}
      onClick={handleClick}
      {...rest}
    >
      <div className="relative h-[2.625rem] w-[2.625rem] rounded-[0.5rem] bg-white">
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
          <span
            className={cn(
              "line-clamp-1 font-sans text-sm font-medium leading-[1.375rem] text-zinc-800 group-disabled:text-zinc-400",
            )}
          >
            {highlightText(beautifyString(title), highlightedText)}
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
        <Plus className="h-5 w-5 text-zinc-50" strokeWidth={2} />
      </div>
    </Button>
  );
};

const IntegrationBlockSkeleton = ({ className }: { className?: string }) => {
  return (
    <Skeleton
      className={cn(
        "flex h-16 w-full min-w-[7.5rem] animate-pulse items-center justify-start gap-3 rounded-[0.75rem] bg-zinc-100 px-[0.875rem] py-[0.625rem]",
        className,
      )}
    >
      <Skeleton className="h-[2.625rem] w-[2.625rem] rounded-[0.5rem] bg-zinc-200" />
      <div className="flex flex-1 flex-col items-start gap-0.5">
        <Skeleton className="h-[1.375rem] w-24 rounded bg-zinc-200" />
        <Skeleton className="h-5 w-32 rounded bg-zinc-200" />
      </div>
      <Skeleton className="h-7 w-7 rounded-[0.5rem] bg-zinc-200" />
    </Skeleton>
  );
};

IntegrationBlock.Skeleton = IntegrationBlockSkeleton;
