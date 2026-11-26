"use client";

import { Text, type TextProps } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";
import React from "react";
import { ShowMoreTextVariant } from "./helpers";

interface Props {
  children: string;
  previewLimit?: number;
  variant?: ShowMoreTextVariant;
  weight?: TextProps["weight"];
  tone?: TextProps["tone"];
  className?: string;
  toggleClassName?: string;
  defaultExpanded?: boolean;
}

export function ShowMoreText({
  children,
  previewLimit = 100,
  variant = "body",
  weight,
  tone,
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
      <Text
        variant={variant}
        weight={weight}
        tone={tone}
        data-slot="show-more-text"
        className={cn(className)}
      >
        {children}
      </Text>
    );
  }

  return (
    <Text
      variant={variant}
      weight={weight}
      tone={tone}
      data-slot="show-more-text"
      className={cn("flex flex-wrap items-center", className)}
    >
      {displayText}
      {!isExpanded && "..."}
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        data-slot="show-more-text-toggle"
        className={cn(
          "mt-2 flex h-[1rem] w-[1.75rem] items-center justify-center rounded-pill bg-surface-hover pb-2 font-medium text-foreground-base",
          toggleClassName,
        )}
        type="button"
      >
        ...
      </button>
    </Text>
  );
}
