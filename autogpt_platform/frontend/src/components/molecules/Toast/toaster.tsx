"use client";

import { Toaster as SonnerToaster } from "sonner";

// Sonner injects its own unlayered stylesheet, which outranks Tailwind's
// utilities layer, so these classes need the important marker. Colours go on
// the per-type keys only: sonner adds `toast` and the type's key together.
const DEFAULT_COLOURS = "border-primary! bg-primary! text-primary-foreground!";

const toastClassNames = {
  toast: "px-4! py-3! shadow-lg!",
  default: DEFAULT_COLOURS,
  loading: DEFAULT_COLOURS,
  title: "-mb-0.5! text-sm! leading-5.5! font-medium! text-inherit!",
  description: "text-xs! leading-5! text-inherit! opacity-90!",
  info: "border-info! bg-info! text-info-foreground!",
  success: "border-success! bg-success! text-success-foreground!",
  warning: "border-warning! bg-warning! text-foreground!",
  error: "border-destructive! bg-destructive! text-destructive-foreground!",
};

export function Toaster() {
  return (
    <SonnerToaster
      position="bottom-center"
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
