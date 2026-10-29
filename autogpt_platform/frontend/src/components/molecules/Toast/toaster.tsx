"use client";

import { Toaster as SonnerToaster } from "sonner";

// `!` beats Sonner's own styles. The status colours sit behind a
// `[data-type=…]` selector, which Sonner sets on the toast, so they outrank
// the base background whatever order Tailwind emits the classes in.
const toastClassNames = {
  toast: "!px-4 !py-3 !bg-action-hover !border-action-hover",
  title: "!-mb-[0.15rem] !text-body !font-medium !text-action-foreground",
  description: "!text-caption !leading-5 !text-action-foreground/95",
  error: "[&[data-type=error]]:!bg-error [&[data-type=error]]:!border-error",
  success:
    "[&[data-type=success]]:!bg-success [&[data-type=success]]:!border-success",
  warning:
    // oxlint-disable-next-line design-system/no-raw-colors -- orange-600 #E77B00 has no role; warning #F59E0B is a different hue
    "[&[data-type=warning]]:!bg-orange-600 [&[data-type=warning]]:!border-orange-600",
};

export function Toaster() {
  return (
    <SonnerToaster
      position="bottom-center"
      richColors
      closeButton
      toastOptions={{ classNames: toastClassNames }}
      icons={{
        success: null,
        error: null,
        warning: null,
        info: null,
      }}
    />
  );
}
