"use client";

import {
  Toast,
  ToastClose,
  ToastDescription,
  ToastProvider,
  ToastTitle,
  ToastViewport,
} from "@/components/ui/toast";
import { useToast } from "@/components/ui/use-toast";

export function Toaster() {
  const { toasts } = useToast();

  // This neat little feature makes the toaster buggy due to the following issue:
  // https://github.com/radix-ui/primitives/issues/2233
  // TODO: Re-enable when the above issue is fixed:
  // const swipeThreshold = toasts.some((toast) => toast.dismissable === false)
  //   ? Infinity
  //   : undefined;
  const swipeThreshold = undefined;

  return (
    <ToastProvider swipeThreshold={swipeThreshold}>
      {toasts.map(
        ({ id, title, description, action, dismissable, ...props }) => (
          <Toast key={id} {...props}>
            <div className="grid gap-1">
              {title && <ToastTitle>{title}</ToastTitle>}
              {description && (
                <ToastDescription>{description}</ToastDescription>
              )}
            </div>
            {action}
            {dismissable !== false && <ToastClose />}
          </Toast>
        ),
      )}
      <ToastViewport />
    </ToastProvider>
  );
}
