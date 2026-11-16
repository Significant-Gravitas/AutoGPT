import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";
import {
  FEEDBACK_TALLY_FORM_ID,
  getFeedbackHiddenFields,
  loadTallyEmbed,
  showFeedbackLoadError,
} from "./helpers";

export function useTallyPageContext() {
  const [pageUrl, setPageUrl] = useState("");
  const [userAgent, setUserAgent] = useState("");
  const pathname = usePathname();

  useEffect(() => {
    // Set client-side values
    if (typeof window !== "undefined") {
      setPageUrl(window.location.href);
      setUserAgent(window.navigator.userAgent);
    }
  }, [pathname]);

  return { pageUrl, userAgent };
}

export function useTallyFeedback(isAuthenticated: boolean | null) {
  const page = useTallyPageContext();
  const [isFormVisible, setIsFormVisible] = useState(false);
  const [isLoading, setIsLoading] = useState(false);

  function handleFormOpen() {
    setIsFormVisible(true);
  }

  function handleFormClose() {
    setIsFormVisible(false);
  }

  async function openFeedbackForm() {
    if (isLoading) return;
    setIsLoading(true);
    try {
      const tally = await loadTallyEmbed();
      tally.openPopup(FEEDBACK_TALLY_FORM_ID, {
        emoji: { text: "👋", animation: "wave" },
        hiddenFields: getFeedbackHiddenFields({ ...page, isAuthenticated }),
        onOpen: handleFormOpen,
        onClose: handleFormClose,
      });
    } catch {
      showFeedbackLoadError();
    } finally {
      setIsLoading(false);
    }
  }

  return { isFormVisible, isLoading, openFeedbackForm };
}
