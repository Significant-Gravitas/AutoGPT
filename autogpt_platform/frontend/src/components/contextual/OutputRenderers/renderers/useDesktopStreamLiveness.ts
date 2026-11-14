import { DesktopStreamEvent } from "@/services/analytics/posthog-events";
import { capturePostHogEvent } from "@/services/analytics/posthog-capture";
import { useEffect, useState } from "react";

export type StreamLiveness = "checking" | "live" | "ended" | "unknown";

const POLL_MS = 10_000;

/** Whether the box still serves the stream behind an owner-bound preview link.
 *  Every pause of the box (the end of each AutoPilot turn among them) stops
 *  the stream, and the link then answers 410 instead of redirecting to it. */
export function useDesktopStreamLiveness(
  url: string,
  enabled: boolean,
): StreamLiveness {
  const [result, setResult] = useState<{
    url: string;
    liveness: StreamLiveness;
  } | null>(null);

  useEffect(() => {
    if (!enabled) return;
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    let liveSince: number | null = null;

    async function check() {
      const liveness = await probe(url);
      if (cancelled) return;
      setResult({ url, liveness });
      if (liveness === "live") liveSince ??= Date.now();
      if (liveness !== "ended") {
        timer = setTimeout(check, POLL_MS);
        return;
      }
      // Only a stream the person had in front of them, not an old card.
      if (liveSince !== null) {
        capturePostHogEvent(DesktopStreamEvent.DESKTOP_STREAM_ENDED, {
          seconds_live: Math.round((Date.now() - liveSince) / 1000),
        });
      }
    }

    void check();
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [url, enabled]);

  if (!enabled) return "unknown";
  return result?.url === url ? result.liveness : "checking";
}

async function probe(url: string): Promise<StreamLiveness> {
  try {
    const response = await fetch(url, {
      redirect: "manual",
      cache: "no-store",
      credentials: "same-origin",
    });
    // The redirect is to the live stream; a browser hides it as opaque.
    if (
      response.type === "opaqueredirect" ||
      (response.status >= 300 && response.status < 400)
    )
      return "live";
    if (response.status === 404 || response.status === 410) return "ended";
    return "unknown";
  } catch {
    return "unknown";
  }
}
