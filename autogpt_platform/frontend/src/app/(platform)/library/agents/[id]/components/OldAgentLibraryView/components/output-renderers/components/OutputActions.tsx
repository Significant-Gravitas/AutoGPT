"use client";

import React, { useState } from "react";
import { CopyIcon, DownloadIcon, ShareIcon, CheckIcon } from "lucide-react";
import { OutputRenderer, OutputMetadata } from "../types";
import { downloadOutputs, DownloadItem } from "../utils/download";

interface OutputActionsProps {
  items: Array<{
    value: any;
    metadata?: OutputMetadata;
    renderer: OutputRenderer;
  }>;
}

export function OutputActions({ items }: OutputActionsProps) {
  const [copied, setCopied] = useState(false);

  const handleCopyAll = async () => {
    const textContents: string[] = [];

    for (const item of items) {
      const copyContent = item.renderer.getCopyContent(
        item.value,
        item.metadata,
      );
      if (copyContent) {
        textContents.push(copyContent);
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

  const handleShare = () => {
    console.log("Share functionality to be implemented");
  };

  return (
    <div className="flex items-center gap-3">
      <button
        onClick={handleCopyAll}
        className="flex h-10 w-10 items-center justify-center rounded-full border border-gray-300 transition-colors hover:bg-gray-50"
        aria-label="Copy all text outputs"
      >
        {copied ? (
          <CheckIcon className="h-4 w-4 text-green-600" />
        ) : (
          <CopyIcon className="h-4 w-4 cursor-pointer text-neutral-500 hover:text-neutral-700" />
        )}
      </button>

      <button
        onClick={handleShare}
        className="flex h-10 w-10 items-center justify-center rounded-full border border-gray-300 transition-colors hover:bg-gray-50"
        aria-label="Share outputs"
      >
        <ShareIcon className="h-4 w-4 cursor-pointer text-neutral-500 hover:text-neutral-700" />
      </button>

      <button
        onClick={handleDownloadAll}
        className="flex h-10 w-10 items-center justify-center rounded-full border border-gray-300 transition-colors hover:bg-gray-50"
        aria-label="Download outputs"
      >
        <DownloadIcon className="h-4 w-4 cursor-pointer text-neutral-500 hover:text-neutral-700" />
      </button>
    </div>
  );
}
