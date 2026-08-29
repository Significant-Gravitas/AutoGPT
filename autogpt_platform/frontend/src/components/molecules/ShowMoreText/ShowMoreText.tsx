"use client";

import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";
import React from "react";
import { ShowMoreTextVariant } from "./helpers";

interface Props {
  children: string;
  previewLimit?: number;
  variant?: ShowMoreTextVariant;
  className?: string;
  toggleClassName?: string;
  defaultExpanded?: boolean;
}

export function ShowMoreText({
  children,
  previewLimit = 100,
  variant = "body",
  className,
  toggleClassName,
  defaultExpanded = false,
}: Props) {
  const [isExpanded, setIsExpanded] = React.useState(defaultExpanded);

  const shouldTruncate = children.length > previewLimit;
  const previewText = shouldTruncate
    ? children.slice(0, previewLimit)
    : children;
  const displayText = isExpanded ? children : previewText;

  if (!shouldTruncate) {
    return (
      <Text variant={variant} className={cn(className)}>
        {children}
      </Text>
    );
  }

  return (
    <Text
      variant={variant}
      className={cn(
        isExpanded
          ? "flex-end flex flex-wrap items-center"
          : "flex-start flex flex-wrap items-center",
        className,
      )}
    >
      {displayText}
      {!isExpanded && "..."}
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className={cn(
          "mt-2 flex h-[1rem] w-[1.75rem] items-center justify-center rounded-full bg-zinc-100 pb-2 font-medium text-black",
          toggleClassName,
        )}
        type="button"
      >
        ...
      </button>
    </Text>
  );
}
