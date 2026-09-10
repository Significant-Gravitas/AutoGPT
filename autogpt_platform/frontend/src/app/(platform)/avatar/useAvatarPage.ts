"use client";

import { toast } from "@/components/molecules/Toast/use-toast";
import {
  avatarUrlFor,
  configForName,
  randomConfig,
  type AvatarConfig,
  type AvatarStatus,
} from "@/components/molecules/BotAvatar/helpers";
import { useState } from "react";
import {
  downloadNameFor,
  type ExpressionChoice,
  type PreviewSize,
} from "./helpers";

export function useAvatarPage() {
  const [config, setConfig] = useState<AvatarConfig>(() =>
    configForName("Otto"),
  );
  const [status, setStatus] = useState<AvatarStatus>("idle");
  const [expression, setExpression] = useState<ExpressionChoice>("auto");
  const [previewSize, setPreviewSize] = useState<PreviewSize>(160);
  const url = avatarUrlFor(config);

  async function copyUrl() {
    try {
      await navigator.clipboard.writeText(url);
      toast({ title: "Avatar URL copied", variant: "success" });
    } catch {
      toast({
        title: "Couldn't copy the URL",
        description: "Copy it from the field instead.",
        variant: "destructive",
      });
    }
  }

  function downloadSvg() {
    const link = document.createElement("a");
    link.href = url;
    link.download = downloadNameFor(url);
    link.rel = "noopener";
    document.body.appendChild(link);
    link.click();
    link.remove();
  }

  function surprise() {
    setConfig(randomConfig());
  }

  return {
    config,
    setConfig,
    status,
    setStatus,
    expression,
    setExpression,
    previewSize,
    setPreviewSize,
    url,
    copyUrl,
    downloadSvg,
    surprise,
  };
}
