import { useCallback } from "react";

export interface PageContext {
  url: string;
  content: string;
}

/**
 * Hook to capture the current page context (URL + full page content)
 */
export function usePageContext() {
  const capturePageContext = useCallback((): PageContext => {
    if (typeof window === "undefined" || typeof document === "undefined") {
      return { url: "", content: "" };
    }

    const url = window.location.href;

    // Capture full page text content
    // Remove script and style elements, then get text
    const clone = document.cloneNode(true) as Document;
    const scripts = clone.querySelectorAll("script, style, noscript");
    scripts.forEach((el) => el.remove());

    // Get text content from body
    const body = clone.body;
    const content = body?.textContent || body?.innerText || "";

    // Clean up whitespace
    const cleanedContent = content
      .replace(/\s+/g, " ")
      .replace(/\n\s*\n/g, "\n")
      .trim();

    return {
      url,
      content: cleanedContent,
    };
  }, []);

  return { capturePageContext };
}
