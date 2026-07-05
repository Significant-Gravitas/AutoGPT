"use client";

import { toast } from "@/components/molecules/Toast/use-toast";
import { CircleNotchIcon, DownloadSimpleIcon } from "@phosphor-icons/react";
import { useState } from "react";

interface Props {
  sessionId: string;
}

export function DownloadButton({ sessionId }: Props) {
  const [isDownloading, setIsDownloading] = useState(false);

  async function handleDownload() {
    setIsDownloading(true);
    try {
      const response = await fetch(
        `/api/proxy/api/chat/sessions/${sessionId}/sandbox/download`,
      );
      if (!response.ok) {
        throw new Error(`Download failed (${response.status})`);
      }
      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = "workspace.tar.gz";
      document.body.appendChild(anchor);
      anchor.click();
      anchor.remove();
      URL.revokeObjectURL(url);
    } catch (error) {
      toast({
        title: "Download failed",
        description: error instanceof Error ? error.message : undefined,
        variant: "destructive",
      });
    } finally {
      setIsDownloading(false);
    }
  }

  return (
    <button
      type="button"
      aria-label="Download workspace"
      disabled={isDownloading}
      onClick={handleDownload}
      className="rounded p-1 text-zinc-500 transition-colors hover:bg-zinc-100 hover:text-zinc-800 disabled:opacity-50"
    >
      {isDownloading ? (
        <CircleNotchIcon size={16} className="animate-spin" />
      ) : (
        <DownloadSimpleIcon size={16} />
      )}
    </button>
  );
}
