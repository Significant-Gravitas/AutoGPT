"use client";
import { cn } from "@/lib/utils";
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
    if (typeof window === "undefined" || !shouldRender) return;

    // Skip if already loaded
    if (window.turnstile) {
      setLoaded(true);
      return;
    }

    // Create script element
    const script = document.createElement("script");
    script.src =
      "https://challenges.cloudflare.com/turnstile/v0/api.js?render=explicit";
    script.async = true;
    script.defer = true;

    script.onload = () => {
      setLoaded(true);
    };

    script.onerror = () => {
      onError?.(new Error("Failed to load Turnstile script"));
    };

    document.head.appendChild(script);

    return () => {
      if (document.head.contains(script)) {
        document.head.removeChild(script);
      }
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
          onVerify(token);
        },
        "expired-callback": () => {
          onExpire?.();
        },
        "error-callback": () => {
          onError?.(new Error("Turnstile widget encountered an error"));
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
          "error-callback"?: () => void;
          action?: string;
        },
      ) => string;
      reset: (widgetId: string) => void;
      remove: (widgetId: string) => void;
    };
  }
}

export default Turnstile;
