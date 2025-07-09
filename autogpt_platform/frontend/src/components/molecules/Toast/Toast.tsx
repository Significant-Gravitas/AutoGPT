"use client";

import { Toaster } from "sonner";
import { useToast } from "./useToast";

export interface ToastProps {
  position?: "top-left" | "top-right" | "bottom-left" | "bottom-right" | "top-center" | "bottom-center";
  theme?: "light" | "dark" | "system";
  richColors?: boolean;
  expand?: boolean;
  duration?: number;
  closeButton?: boolean;
  visibleToasts?: number;
  style?: React.CSSProperties;
  className?: string;
  toastOptions?: {
    style?: React.CSSProperties;
    className?: string;
    duration?: number;
    unstyled?: boolean;
  };
}

export function Toast({
  position = "top-right",
  theme = "system",
  richColors = true,
  expand = false,
  duration = 4000,
  closeButton = false,
  visibleToasts = 3,
  style,
  className,
  toastOptions,
}: ToastProps) {
  const { toastConfig } = useToast({
    position,
    theme,
    richColors,
    expand,
    duration,
    closeButton,
    visibleToasts,
    style,
    className,
    toastOptions,
  });

  return <Toaster {...toastConfig} />;
}

Toast.displayName = "Toast";