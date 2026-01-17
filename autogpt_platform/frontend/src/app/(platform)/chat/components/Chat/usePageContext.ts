import { useCallback } from "react";

export interface PageContext {
  url: string;
  content: string;
}

const MAX_CONTENT_CHARS = 10000;

/**
 * Hook to capture the current page context (URL + full page content)
 * Privacy-hardened: removes sensitive inputs and enforces content size limits
 */
export function usePageContext() {
  const capturePageContext = useCallback((): PageContext => {
    if (typeof window === "undefined" || typeof document === "undefined") {
      return { url: "", content: "" };
    }

    const url = window.location.href;

    // Clone document to avoid modifying the original
    const clone = document.cloneNode(true) as Document;

    // Remove script, style, and noscript elements
    const scripts = clone.querySelectorAll("script, style, noscript");
    scripts.forEach((el) => el.remove());

    // Remove sensitive elements and their content
    const sensitiveSelectors = [
      "input",
      "textarea",
      "[contenteditable]",
      'input[type="password"]',
      'input[type="email"]',
      'input[type="tel"]',
      'input[type="search"]',
      'input[type="hidden"]',
      "form",
      "[data-sensitive]",
      "[data-sensitive='true']",
    ];

    sensitiveSelectors.forEach((selector) => {
      const elements = clone.querySelectorAll(selector);
      elements.forEach((el) => {
        // For form elements, remove the entire element
        if (el.tagName === "FORM") {
          el.remove();
        } else {
          // For inputs and textareas, clear their values but keep the element structure
          if (
            el instanceof HTMLInputElement ||
            el instanceof HTMLTextAreaElement
          ) {
            el.value = "";
            el.textContent = "";
          } else {
            // For other sensitive elements, remove them entirely
            el.remove();
          }
        }
      });
    });

    // Strip any remaining input values that might have been missed
    const allInputs = clone.querySelectorAll("input, textarea");
    allInputs.forEach((el) => {
      if (el instanceof HTMLInputElement || el instanceof HTMLTextAreaElement) {
        el.value = "";
        el.textContent = "";
      }
    });

    // Get text content from body
    const body = clone.body;
    const content = body?.textContent || body?.innerText || "";

    // Clean up whitespace
    let cleanedContent = content
      .replace(/\s+/g, " ")
      .replace(/\n\s*\n/g, "\n")
      .trim();

    // Enforce maximum content size
    if (cleanedContent.length > MAX_CONTENT_CHARS) {
      cleanedContent =
        cleanedContent.substring(0, MAX_CONTENT_CHARS) + "... [truncated]";
    }

    return {
      url,
      content: cleanedContent,
    };
  }, []);

  return { capturePageContext };
}
