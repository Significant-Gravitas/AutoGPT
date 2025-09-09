"use client";

import React, { useState } from "react";
import { CopyIcon, CheckIcon } from "lucide-react";
import { OutputRenderer, OutputMetadata } from "../types";
import { copyToClipboard } from "../utils/copy";

interface OutputItemProps {
  value: any;
  metadata?: OutputMetadata;
  renderer: OutputRenderer;
  label?: string;
}

export function OutputItem({
  value,
  metadata,
  renderer,
  label,
}: OutputItemProps) {
  const [showCopyButton, setShowCopyButton] = useState(false);
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    const copyContent = renderer.getCopyContent(value, metadata);
    if (copyContent) {
      try {
        await copyToClipboard(copyContent);
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
      } catch (error) {
        console.error("Failed to copy:", error);
      }
    }
  };

  const canCopy = renderer.getCopyContent(value, metadata) !== null;

  return (
    <div
      className="relative"
      onMouseEnter={() => setShowCopyButton(true)}
      onMouseLeave={() => setShowCopyButton(false)}
    >
      {label && (
        <label className="mb-1.5 block text-sm font-medium">{label}</label>
      )}

      <div className="relative">
        {renderer.render(value, metadata)}

        {canCopy && showCopyButton && (
          <button
            onClick={handleCopy}
            className="absolute right-2 top-2 rounded-md border border-gray-200 bg-background/80 p-1.5 backdrop-blur-sm transition-all duration-200 hover:bg-gray-100"
            aria-label="Copy content"
          >
            {copied ? (
              <CheckIcon className="h-4 w-4 text-green-600" />
            ) : (
              <CopyIcon className="h-4 w-4 text-gray-600" />
            )}
          </button>
        )}
      </div>
    </div>
  );
}
