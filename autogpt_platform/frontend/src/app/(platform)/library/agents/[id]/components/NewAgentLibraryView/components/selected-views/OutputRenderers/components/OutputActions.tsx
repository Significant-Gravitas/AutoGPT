"use client";

import React, { useState } from "react";
import { CheckIcon, CopyIcon, DownloadIcon } from "@phosphor-icons/react";
import { Button } from "@/components/atoms/Button/Button";
import { OutputRenderer, OutputMetadata } from "../types";
import { downloadOutputs } from "../utils/download";
import { cn } from "@/lib/utils";

interface OutputActionsProps {
  items: Array<{
    value: unknown;
    metadata?: OutputMetadata;
    renderer: OutputRenderer;
  }>;
  isPrimary?: boolean;
  className?: string;
}

export function OutputActions({
  items,
  isPrimary = false,
}: OutputActionsProps) {
  const [copied, setCopied] = useState(false);

  const handleCopyAll = async () => {
    const textContents: string[] = [];

    for (const item of items) {
      const copyContent = item.renderer.getCopyContent(
        item.value,
        item.metadata,
      );
      if (
        copyContent &&
        item.renderer.isConcatenable(item.value, item.metadata)
      ) {
        // For concatenable items, extract the text
        let text: string;
        if (typeof copyContent.data === "string") {
          text = copyContent.data;
        } else if (copyContent.fallbackText) {
          text = copyContent.fallbackText;
        } else {
          continue;
        }
        textContents.push(text);
      }
    }

    if (textContents.length > 0) {
      const combinedText = textContents.join("\n\n");
      try {
        await navigator.clipboard.writeText(combinedText);
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
      } catch (error) {
        console.error("Failed to copy:", error);
      }
    }
  };

  const handleDownloadAll = () => {
    downloadOutputs(items);
  };

  return (
    <div className="flex items-center gap-3">
      <Button
        variant={isPrimary ? "primary" : "ghost"}
        size={isPrimary ? "small" : "icon"}
        onClick={handleCopyAll}
        aria-label="Copy all text outputs"
        className={cn(isPrimary ? "min-w-0" : "")}
      >
        {copied ? (
          <CheckIcon className="size-4 text-green-600" />
        ) : (
          <CopyIcon
            className={cn(
              "size-4",
              isPrimary ? "text-white" : "text-neutral-500",
            )}
          />
        )}
        {isPrimary && <span>Copy All</span>}
      </Button>

      <Button
        variant={isPrimary ? "primary" : "ghost"}
        size={isPrimary ? "small" : "icon"}
        onClick={handleDownloadAll}
        aria-label="Download outputs"
        className={cn(isPrimary ? "min-w-0" : "")}
      >
        <DownloadIcon
          className={cn(
            "size-4",
            isPrimary ? "text-white" : "text-neutral-500",
          )}
        />
        {isPrimary && <span>Download All</span>}
      </Button>
    </div>
  );
}
