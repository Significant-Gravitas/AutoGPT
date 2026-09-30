export const DATA_LAYER_NAME = "dataLayer";

declare global {
  interface Window {
    // The tag's own command shim, defined by the init script in SetupAnalytics.
    // gtag.js only executes dataLayer entries that are real `arguments`
    // objects, which only that shim can produce; everything routes through it.
    gtag?: (...args: unknown[]) => void;
  }
}

export function gtag(...args: unknown[]): boolean {
  if (typeof window === "undefined") return false;
  const tag = window.gtag;
  if (typeof tag !== "function") return false;
  tag(...args);
  return true;
}
