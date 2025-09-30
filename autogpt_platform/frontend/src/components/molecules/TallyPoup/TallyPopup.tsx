"use client";

import React, { useEffect, useState } from "react";
import { Button } from "../../__legacy__/ui/button";
import { QuestionMarkCircledIcon } from "@radix-ui/react-icons";
import { useRouter, usePathname } from "next/navigation";
import * as Sentry from "@sentry/nextjs";
import { getCurrentUser } from "@/lib/supabase/actions";

const TallyPopupSimple = () => {
  const [isFormVisible, setIsFormVisible] = useState(false);
  const [sentryReplayId, setSentryReplayId] = useState("");
  const [replayUrl, setReplayUrl] = useState("");
  const [pageUrl, setPageUrl] = useState("");
  const [userAgent, setUserAgent] = useState("");
  const [isAuthenticated, setIsAuthenticated] = useState<boolean | null>(null);
  // const [userId, setUserId] = useState<string>("");
  const [userEmail, setUserEmail] = useState<string>("");
  const router = useRouter();
  const pathname = usePathname();

  const [show_tutorial, setShowTutorial] = useState(false);

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
      // setUserId(user?.id || "");
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

  if (isFormVisible) {
    return null;
  }

  const resetTutorial = () => {
    router.push("/build?resetTutorial=true");
  };

  return (
    <div className="fixed bottom-1 right-24 z-20 hidden select-none items-center gap-4 p-3 transition-all duration-300 ease-in-out md:flex">
      {show_tutorial && (
        <Button
          variant="default"
          onClick={resetTutorial}
          className="mb-0 h-14 w-28 rounded-2xl bg-[rgba(65,65,64,1)] text-left font-sans text-lg font-medium leading-6"
        >
          Tutorial
        </Button>
      )}
      <Button
        className="h-14 w-14 rounded-full bg-[rgba(65,65,64,1)]"
        variant="default"
        data-tally-open="3yx2L0"
        data-tally-emoji-text="👋"
        data-tally-emoji-animation="wave"
        data-sentry-replay-id={sentryReplayId || "not-initialized"}
        data-sentry-replay-url={replayUrl || "not-initialized"}
        data-user-agent={userAgent}
        data-page-url={pageUrl}
        data-is-authenticated={
          isAuthenticated === null ? "unknown" : String(isAuthenticated)
        }
        data-email={userEmail || "not-authenticated"}
        // data-user-id={userId || "not-authenticated"}
      >
        <QuestionMarkCircledIcon className="h-14 w-14" />
        <span className="sr-only">Reach Out</span>
      </Button>
    </div>
  );
};

export default TallyPopupSimple;
