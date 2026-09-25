"use client";

import { toast } from "@/components/molecules/Toast/use-toast";
import { useEffect, useRef, useState } from "react";

export function useArtifactFullscreen() {
  const fullscreenRef = useRef<HTMLDivElement>(null);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [canFullscreen, setCanFullscreen] = useState(false);

  useEffect(() => {
    setCanFullscreen(!!document.fullscreenEnabled);
    function handleFullscreenChange() {
      setIsFullscreen(
        !!fullscreenRef.current &&
          document.fullscreenElement === fullscreenRef.current,
      );
    }
    document.addEventListener("fullscreenchange", handleFullscreenChange);
    return () => {
      document.removeEventListener("fullscreenchange", handleFullscreenChange);
    };
  }, []);

  async function toggleFullscreen() {
    if (!fullscreenRef.current) return;
    try {
      if (document.fullscreenElement === fullscreenRef.current) {
        await document.exitFullscreen();
      } else {
        await fullscreenRef.current?.requestFullscreen();
      }
    } catch {
      toast({
        title: "Couldn't change fullscreen mode",
        variant: "destructive",
      });
    }
  }

  return { fullscreenRef, isFullscreen, canFullscreen, toggleFullscreen };
}
