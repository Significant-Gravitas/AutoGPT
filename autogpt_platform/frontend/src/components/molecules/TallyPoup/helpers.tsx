import { toast } from "@/components/molecules/Toast/use-toast";

const TALLY_EMBED_URL = "https://tally.so/widgets/embed.js";
export const FEEDBACK_TALLY_FORM_ID = "3yx2L0";
const FEEDBACK_FORM_URL = `https://tally.so/r/${FEEDBACK_TALLY_FORM_ID}`;

interface TallyPopupOptions {
  emoji?: { text: string; animation: string };
  hiddenFields?: Record<string, string>;
  onOpen?: () => void;
  onClose?: () => void;
}

interface TallyApi {
  openPopup: (formId: string, options?: TallyPopupOptions) => void;
}

declare global {
  interface Window {
    Tally?: TallyApi;
  }
}

let pendingLoad: Promise<TallyApi> | null = null;

export function loadTallyEmbed(): Promise<TallyApi> {
  if (window.Tally) return Promise.resolve(window.Tally);
  if (pendingLoad) return pendingLoad;

  pendingLoad = new Promise<TallyApi>((resolve, reject) => {
    const script = document.createElement("script");
    script.src = TALLY_EMBED_URL;
    script.async = true;

    script.addEventListener("load", () => {
      pendingLoad = null;
      if (window.Tally) {
        resolve(window.Tally);
      } else {
        script.remove();
        reject(new Error("Tally script loaded without its API"));
      }
    });
    script.addEventListener("error", () => {
      pendingLoad = null;
      script.remove();
      reject(new Error("Tally script failed to load"));
    });

    document.head.appendChild(script);
  });

  return pendingLoad;
}

interface FeedbackContext {
  sentryReplayId: string;
  replayUrl: string;
  pageUrl: string;
  isAuthenticated: boolean | null;
}

export function getFeedbackHiddenFields(context: FeedbackContext) {
  return {
    sentryReplayId: context.sentryReplayId || "not-initialized",
    sentryReplayUrl: context.replayUrl || "not-initialized",
    pageUrl: context.pageUrl
      ? context.pageUrl.split("?")[0]
      : "not-initialized",
    isAuthenticated:
      context.isAuthenticated === null
        ? "unknown"
        : String(context.isAuthenticated),
  };
}

export function showFeedbackLoadError() {
  toast({
    title: "Couldn't open the feedback form",
    description: (
      <>
        It may be blocked by your browser.{" "}
        <a
          href={FEEDBACK_FORM_URL}
          target="_blank"
          rel="noopener noreferrer"
          className="underline"
        >
          Open the form in a new tab
        </a>
      </>
    ),
    variant: "destructive",
    duration: 10000,
  });
}
