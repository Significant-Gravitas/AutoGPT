"use client";

import { Toaster as SonnerToaster } from "sonner";

// `!` beats Sonner's own styles. The status colours sit behind a
// `[data-type=…]` selector, which Sonner sets on the toast, so they outrank
// the base background whatever order Tailwind emits the classes in.
// TODO(tokens): no role for green-400 (#45B56E), orange-600 (#E77B00),
// red-400 (#F26969) or the description's #F4F4F5.
const toastClassNames = {
  toast: "!px-4 !py-[0.8rem] !bg-action-hover !border-action-hover",
  title: "!-mb-[0.15rem] !text-body !font-medium !text-action-foreground",
  description: "!text-caption !leading-5 !text-[#f4f4f5]",
  error:
    "[&[data-type=error]]:!bg-red-400 [&[data-type=error]]:!border-red-400",
  success:
    "[&[data-type=success]]:!bg-green-400 [&[data-type=success]]:!border-green-400",
  warning:
    "[&[data-type=warning]]:!bg-orange-600 [&[data-type=warning]]:!border-orange-600",
};

export function Toaster() {
  return (
    <SonnerToaster
      position="bottom-center"
      richColors
      closeButton
      toastOptions={{ classNames: toastClassNames }}
      className="custom__toast"
      icons={{
        success: null,
        error: null,
        warning: null,
        info: null,
      }}
    />
  );
}
