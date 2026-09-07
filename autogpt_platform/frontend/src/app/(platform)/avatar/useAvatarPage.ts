import {
  decodeConfig,
  DEFAULT_CONFIG,
  encodeConfig,
  randomConfig,
  type AccessoryId,
  type AvatarConfig,
  type AvatarStatus,
  type ColorId,
  type ShapeId,
} from "@/components/molecules/BotAvatar/helpers";
import { toast } from "@/components/molecules/Toast/use-toast";
import { useQueryState } from "nuqs";
import { useRef, useState } from "react";
import {
  downloadBlob,
  exportFilename,
  QUERY_KEY,
  serializeSvg,
  svgToPngBlob,
} from "./helpers";

export function useAvatarPage() {
  const [encoded, setEncoded] = useQueryState(QUERY_KEY);
  const [status, setStatus] = useState<AvatarStatus>("idle");
  const [isExporting, setIsExporting] = useState(false);
  const exportRef = useRef<HTMLDivElement>(null);
  const config = decodeConfig(encoded);

  function update(patch: Partial<AvatarConfig>) {
    setEncoded(encodeConfig({ ...config, ...patch }));
  }

  function setShape(shape: ShapeId) {
    update({ shape });
  }

  function setColor(color: ColorId) {
    update({ color });
  }

  function setAccessory(accessory: AccessoryId) {
    update({ accessory });
  }

  function shuffle() {
    setEncoded(encodeConfig(randomConfig()));
  }

  function reset() {
    setEncoded(encodeConfig(DEFAULT_CONFIG));
    setStatus("idle");
  }

  async function copyLink() {
    try {
      await navigator.clipboard.writeText(window.location.href);
      toast({ title: "Link copied" });
    } catch {
      toast({ title: "Couldn't copy the link", variant: "destructive" });
    }
  }

  function exportSvgMarkup() {
    const svg = exportRef.current?.querySelector("svg");
    return svg ? serializeSvg(svg) : null;
  }

  function downloadSvg() {
    const markup = exportSvgMarkup();
    if (!markup) return;
    downloadBlob(
      new Blob([markup], { type: "image/svg+xml" }),
      exportFilename(config, "svg"),
    );
  }

  async function downloadPng() {
    const markup = exportSvgMarkup();
    if (!markup) return;
    setIsExporting(true);
    try {
      downloadBlob(await svgToPngBlob(markup), exportFilename(config, "png"));
    } catch {
      toast({ title: "Couldn't render the PNG", variant: "destructive" });
    } finally {
      setIsExporting(false);
    }
  }

  return {
    config,
    status,
    setStatus,
    setShape,
    setColor,
    setAccessory,
    shuffle,
    reset,
    copyLink,
    downloadSvg,
    downloadPng,
    isExporting,
    exportRef,
  };
}
