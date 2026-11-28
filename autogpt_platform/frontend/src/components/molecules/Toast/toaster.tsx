"use client";

import { Toaster as SonnerToaster } from "sonner";

// `!` beats Sonner's own styles. The status colours sit behind a
// `[data-type=…]` selector, which Sonner sets on the toast, so they outrank
// the base background and text whatever order Tailwind emits the classes in.
// A status toast is a saturated fill, so its text is on-color; the default
// toast is the inverse surface, with inverse text.
const toastClassNames = {
  toast: "!px-4 !py-3 !bg-surface-inverse-hover !border-surface-inverse-hover",
  title: "!-mb-[0.15rem] !text-body !font-medium !text-foreground-inverse",
  description: "!text-caption !leading-5 !text-foreground-inverse/95",
  error:
    "[&[data-type=error]]:!bg-error [&[data-type=error]]:!border-error [&[data-type=error]_[data-title]]:!text-foreground-on-color [&[data-type=error]_[data-description]]:!text-foreground-on-color/95",
  success:
    "[&[data-type=success]]:!bg-success [&[data-type=success]]:!border-success [&[data-type=success]_[data-title]]:!text-foreground-on-color [&[data-type=success]_[data-description]]:!text-foreground-on-color/95",
  warning:
    "[&[data-type=warning]]:!bg-orange-foreground [&[data-type=warning]]:!border-orange-foreground [&[data-type=warning]_[data-title]]:!text-foreground-on-color [&[data-type=warning]_[data-description]]:!text-foreground-on-color/95",
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
