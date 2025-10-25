import { usePathname, useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import * as Sentry from "@sentry/nextjs";
import { getCurrentUser } from "@/lib/supabase/actions";

export function useTallyPopup() {
  const [isFormVisible, setIsFormVisible] = useState(false);
  const [sentryReplayId, setSentryReplayId] = useState("");
  const [replayUrl, setReplayUrl] = useState("");
  const [pageUrl, setPageUrl] = useState("");
  const [userAgent, setUserAgent] = useState("");
  const [isAuthenticated, setIsAuthenticated] = useState<boolean | null>(null);
  const [userEmail, setUserEmail] = useState<string>("");
  const router = useRouter();
  const pathname = usePathname();

  const [showTutorial, setShowTutorial] = useState(false);

  useEffect(() => {
    setShowTutorial(pathname.includes("build"));
  }, [pathname]);

  useEffect(() => {
    // Set client-side values
    if (typeof window !== "undefined") {
      setPageUrl(window.location.href);
      setUserAgent(window.navigator.userAgent);

      const replay = Sentry.getReplay();

      if (replay) {
        const replayId = replay.getReplayId();

        if (replayId) {
          setSentryReplayId(replayId);
          const orgSlug = "significant-gravitas";
          setReplayUrl(`https://${orgSlug}.sentry.io/replays/${replayId}/`);
        }
      }
    }
  }, [pathname]);

  useEffect(() => {
    // Check authentication status using server action (works with httpOnly cookies)
    getCurrentUser().then(({ user }) => {
      setIsAuthenticated(user != null);
      setUserEmail(user?.email || "");
    });
  }, [pathname]);

  useEffect(() => {
    // Load Tally script
    const script = document.createElement("script");
    script.src = "https://tally.so/widgets/embed.js";
    script.async = true;
    document.head.appendChild(script);

    // Setup event listeners for Tally events
    const handleTallyMessage = (event: MessageEvent) => {
      if (typeof event.data === "string") {
        // Ignore iframe-resizer messages
        if (
          event.data.startsWith("[iFrameSize") ||
          event.data.startsWith("[iFrameResizer")
        ) {
          return;
        }

        try {
          const data = JSON.parse(event.data);

          // Only process Tally events
          if (!data.event?.startsWith("Tally.")) {
            return;
          }

          if (data.event === "Tally.FormLoaded") {
            setIsFormVisible(true);

            // Flush Sentry replay when form opens
            if (typeof window !== "undefined") {
              const replay = Sentry.getReplay();
              if (replay) {
                replay.flush();
              }
            }
          } else if (data.event === "Tally.PopupClosed") {
            setIsFormVisible(false);
          }
        } catch (error) {
          // Only log errors for messages we care about
          if (event.data.includes("Tally")) {
            console.error("Error parsing Tally message:", error);
          }
        }
      }
    };

    window.addEventListener("message", handleTallyMessage);

    return () => {
      document.head.removeChild(script);
      window.removeEventListener("message", handleTallyMessage);
    };
  }, []);

  function handleResetTutorial() {
    router.push("/build?resetTutorial=true");
  }

  return {
    state: {
      showTutorial,
      sentryReplayId,
      replayUrl,
      pageUrl,
      userAgent,
      isAuthenticated,
      isFormVisible,
      userEmail,
    },
    handlers: {
      handleResetTutorial,
    },
  };
}
