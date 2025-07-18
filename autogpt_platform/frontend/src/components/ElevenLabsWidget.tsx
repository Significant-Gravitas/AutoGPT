"use client";

import { useEffect, createElement } from "react";
import { getBehaveAs, BehaveAs } from "@/lib/utils";

export default function ElevenLabsWidget() {
  const behaveAs = getBehaveAs();

  useEffect(() => {
    if (behaveAs !== BehaveAs.CLOUD) return;

    // Check if script is already loaded
    const existingScript = document.querySelector(
      'script[src="https://unpkg.com/@elevenlabs/convai-widget-embed"]',
    );

    if (existingScript) return;

    const script = document.createElement("script");
    script.src = "https://unpkg.com/@elevenlabs/convai-widget-embed";
    script.async = true;
    script.type = "text/javascript";

    document.head.appendChild(script);

    return () => {
      const scriptToRemove = document.querySelector(
        'script[src="https://unpkg.com/@elevenlabs/convai-widget-embed"]',
      );
      if (scriptToRemove) {
        scriptToRemove.remove();
      }
    };
  }, [behaveAs]);

  if (behaveAs !== BehaveAs.CLOUD) return null;

  return createElement("elevenlabs-convai", {
    "agent-id": "agent_01k0catqvjef0sk50r03cj49ek",
  });
}
