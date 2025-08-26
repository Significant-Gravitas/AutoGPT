"use client";

import React from "react";
import { cn } from "@/lib/utils";
import { CaretDownIcon, CaretUpIcon } from "@phosphor-icons/react";
import { Text } from "@/components/atoms/Text/Text";
import { getIconSize, ShowMoreTextVariant } from "./helpers";

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
            <CaretUpIcon size={iconSize} weight="bold" />
            <span>less</span>
          </>
        ) : (
          <>
            <CaretDownIcon size={iconSize} weight="bold" />
            <span>more</span>
          </>
        )}
      </button>
    </Text>
  );
}

export default ShowMore;
