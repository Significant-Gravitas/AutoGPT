"use client";
import { cn } from "@/lib/utils";
import { isServerSide } from "@/lib/utils/is-server-side";
import { useEffect, useRef, useState } from "react";

export interface TurnstileProps {
  siteKey: string;
  onVerify: (token: string) => void;
  onExpire?: () => void;
  onError?: (error: Error) => void;
  action?: string;
  className?: string;
  id?: string;
  shouldRender?: boolean;
  setWidgetId?: (id: string | null) => void;
}

export function Turnstile({
  siteKey,
  onVerify,
  onExpire,
  onError,
  action,
  className,
  id = "cf-turnstile",
  shouldRender = true,
  setWidgetId,
}: TurnstileProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const widgetIdRef = useRef<string | null>(null);
  const [loaded, setLoaded] = useState(false);

  // Load the Turnstile script
  useEffect(() => {
    if (isServerSide() || !shouldRender) return;

    // Skip if already loaded
    if (window.turnstile) {
      setLoaded(true);
      return;
    }

    const scriptSrc =
      "https://challenges.cloudflare.com/turnstile/v0/api.js?render=explicit";

    // If a script already exists, reuse it and attach listeners
    const existingScript = Array.from(document.scripts).find(
      (s) => s.src === scriptSrc,
    );

    if (existingScript) {
      if (window.turnstile) {
        setLoaded(true);
        return;
      }

      const handleLoad: EventListener = () => {
        setLoaded(true);
      };
      const handleError: EventListener = () => {
        onError?.(new Error("Failed to load Turnstile script"));
      };

      existingScript.addEventListener("load", handleLoad);
      existingScript.addEventListener("error", handleError);

      return () => {
        existingScript.removeEventListener("load", handleLoad);
        existingScript.removeEventListener("error", handleError);
      };
    }

    // Create a single script element if not present and keep it in the document
    const script = document.createElement("script");
    script.src = scriptSrc;
    script.async = true;
    script.defer = true;

    const handleLoad: EventListener = () => {
      setLoaded(true);
    };
    const handleError: EventListener = () => {
      onError?.(new Error("Failed to load Turnstile script"));
    };

    script.addEventListener("load", handleLoad);
    script.addEventListener("error", handleError);

    document.head.appendChild(script);

    return () => {
      script.removeEventListener("load", handleLoad);
      script.removeEventListener("error", handleError);
    };
  }, [onError, shouldRender]);

  // Initialize and render the widget when script is loaded
  useEffect(() => {
    if (!loaded || !containerRef.current || !window.turnstile || !shouldRender)
      return;

    // Reset any existing widget
    if (widgetIdRef.current && window.turnstile) {
      try {
        window.turnstile.reset(widgetIdRef.current);
      } catch (err) {
        console.warn("Failed to reset existing Turnstile widget:", err);
      }
    }

    // Render a new widget
    if (window.turnstile) {
      widgetIdRef.current = window.turnstile.render(containerRef.current, {
        sitekey: siteKey,
        callback: (token: string) => {
          console.log("[Turnstile] Verification successful");
          onVerify(token);
        },
        "expired-callback": () => {
          console.warn("[Turnstile] Token expired");
          onExpire?.();
        },
        "error-callback": (errorCode: string) => {
          // Capture the actual Cloudflare error code for debugging
          console.error("[Turnstile] Error occurred:", errorCode);

          // Map Cloudflare error codes to user-friendly messages
          const errorMessages: Record<string, string> = {
            "110100": "Invalid site configuration. Please contact support.",
            "110200": "Domain not allowed for this CAPTCHA.",
            "110201": "Hostname mismatch. Please try refreshing the page.",
            "110400": "Invalid verification action.",
            "110500": "Invalid custom data.",
            "110600": "JavaScript execution error.",
            "300010": "Session timeout. Please try again.",
            "300020": "Network error. Please check your connection.",
            "300030": "Challenge closed by user.",
            "600010": "Verification timeout. Please try again.",
            "600020": "Internal error. Please try again later.",
          };

          const errorMessage =
            errorMessages[errorCode] ||
            `Verification failed (Error: ${errorCode}). Please try again.`;

          const error = new Error(errorMessage);
          (error as any).errorCode = errorCode;
          onError?.(error);
        },
        action,
      });

      // Notify the hook about the widget ID
      setWidgetId?.(widgetIdRef.current);
    }

    return () => {
      if (widgetIdRef.current && window.turnstile) {
        try {
          window.turnstile.remove(widgetIdRef.current);
        } catch (err) {
          console.warn("Failed to remove Turnstile widget:", err);
        }
        setWidgetId?.(null);
        widgetIdRef.current = null;
      }
    };
  }, [
    loaded,
    siteKey,
    onVerify,
    onExpire,
    onError,
    action,
    shouldRender,
    setWidgetId,
  ]);

  // Method to reset the widget manually
  useEffect(() => {
    if (loaded && widgetIdRef.current && window.turnstile && shouldRender) {
      window.turnstile.reset(widgetIdRef.current);
    }
  }, [loaded, shouldRender]);

  // If shouldRender is false, don't render anything
  if (!shouldRender) {
    return null;
  }

  return (
    <div
      id={id}
      ref={containerRef}
      className={cn("my-4 flex items-center justify-center", className)}
    />
  );
}

// Add TypeScript interface to Window to include turnstile property
declare global {
  interface Window {
    turnstile?: {
      render: (
        container: HTMLElement,
        options: {
          sitekey: string;
          callback: (token: string) => void;
          "expired-callback"?: () => void;
          "error-callback"?: (errorCode: string) => void;
          action?: string;
        },
      ) => string;
      reset: (widgetId: string) => void;
      remove: (widgetId: string) => void;
      getResponse: (widgetId: string) => string | undefined;
    };
  }
}

export default Turnstile;
