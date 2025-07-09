import { useMemo } from "react";
import { toast } from "sonner";
import { getToastConfig } from "./helpers";
import type { ToastProps } from "./Toast";

export interface UseToastParams extends ToastProps {}

export interface UseToastReturn {
  toastConfig: {
    position: ToastProps["position"];
    theme: ToastProps["theme"];
    richColors: boolean;
    expand: boolean;
    duration: number;
    closeButton: boolean;
    visibleToasts: number;
    style?: React.CSSProperties;
    className?: string;
    toastOptions?: ToastProps["toastOptions"];
  };
  showToast: {
    success: typeof toast.success;
    error: typeof toast.error;
    info: typeof toast.info;
    warning: typeof toast.warning;
    loading: typeof toast.loading;
    promise: typeof toast.promise;
    custom: typeof toast.custom;
    dismiss: typeof toast.dismiss;
  };
}

export function useToast(params: UseToastParams): UseToastReturn {
  const toastConfig = useMemo(() => getToastConfig(params), [
    params.position,
    params.theme,
    params.richColors,
    params.expand,
    params.duration,
    params.closeButton,
    params.visibleToasts,
    params.style,
    params.className,
    params.toastOptions,
  ]);

  const showToast = useMemo(
    () => ({
      success: toast.success,
      error: toast.error,
      info: toast.info,
      warning: toast.warning,
      loading: toast.loading,
      promise: toast.promise,
      custom: toast.custom,
      dismiss: toast.dismiss,
    }),
    [],
  );

  return {
    toastConfig,
    showToast,
  };
}