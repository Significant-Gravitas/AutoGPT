"use client";

import React from "react";
import { cn } from "@/lib/utils";
import { Text } from "@/components/atoms/Text/Text";
import { getIconSize, ShowMoreTextVariant } from "./helpers";
import { ArrowDown01Icon, ArrowUp01Icon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";

interface ShowMoreProps {
  children: string;
  previewLimit?: number;
  variant?: ShowMoreTextVariant;
  className?: string;
  toggleClassName?: string;
  defaultExpanded?: boolean;
}

export function ShowMore({
  children,
  previewLimit = 100,
  variant = "body",
  className,
  toggleClassName,
  defaultExpanded = false,
}: ShowMoreProps) {
  const [isExpanded, setIsExpanded] = React.useState(defaultExpanded);

  const shouldTruncate = children.length > previewLimit;
  const previewText = shouldTruncate
    ? children.slice(0, previewLimit)
    : children;
  const displayText = isExpanded ? children : previewText;
  const iconSize = getIconSize(variant);

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
          "ml-1 inline-flex items-center gap-1 font-medium text-black",
          toggleClassName,
        )}
        type="button"
      >
        {isExpanded ? (
          <>
            <Icon icon={ArrowUp01Icon} size={iconSize} />
            <span>less</span>
          </>
        ) : (
          <>
            <Icon icon={ArrowDown01Icon} size={iconSize} />
            <span>more</span>
          </>
        )}
      </button>
    </Text>
  );
}

export default ShowMore;
