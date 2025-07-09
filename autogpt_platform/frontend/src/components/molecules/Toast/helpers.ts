import type { ToastProps } from "./Toast";

export function getToastConfig(params: ToastProps) {
  return {
    position: params.position || "top-right",
    theme: params.theme || "system",
    richColors: params.richColors !== undefined ? params.richColors : true,
    expand: params.expand !== undefined ? params.expand : false,
    duration: params.duration || 4000,
    closeButton: params.closeButton !== undefined ? params.closeButton : false,
    visibleToasts: params.visibleToasts || 3,
    style: params.style,
    className: params.className,
    toastOptions: params.toastOptions,
  };
}

export function createToastMessage(
  message: string,
  type: "success" | "error" | "info" | "warning" | "loading" = "info",
) {
  const iconMap = {
    success: "✅",
    error: "❌",
    info: "ℹ️",
    warning: "⚠️",
    loading: "⏳",
  };

  return {
    message,
    icon: iconMap[type],
    type,
  };
}

export function validateToastOptions(options: ToastProps["toastOptions"]) {
  if (!options) return true;
  
  if (options.duration && options.duration < 0) {
    console.warn("Toast duration should be positive");
    return false;
  }
  
  return true;
}