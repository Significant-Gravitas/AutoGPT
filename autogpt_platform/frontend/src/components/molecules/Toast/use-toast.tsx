"use client";

import * as React from "react";
import { toast as sonnerToast } from "sonner";

export interface ToastProps {
  title?: React.ReactNode;
  description?: React.ReactNode;
  variant?: "default" | "destructive" | "success" | "info";
  duration?: number;
  action?: React.ReactNode;
  dismissable?: boolean;
}

type ToasterToast = ToastProps & {
  id: string;
  open?: boolean;
  onOpenChange?: (open: boolean) => void;
};

interface State {
  toasts: ToasterToast[];
}

type Toast = Omit<ToasterToast, "id">;

function toast({
  title,
  description,
  variant = "default",
  duration = 5000,
  action,
  dismissable = true,
  ..._props
}: Toast) {
  const message = title || description || "";
  const descriptionText = title && description ? description : undefined;

  const toastOptions = {
    duration: dismissable ? duration : Infinity,
    action,
    description: descriptionText,
  };

  let toastId: string | number;

  switch (variant) {
    case "destructive":
      toastId = sonnerToast.error(message, toastOptions);
      break;
    case "success":
      toastId = sonnerToast.success(message, toastOptions);
      break;
    case "info":
      toastId = sonnerToast.info(message, toastOptions);
      break;
    default:
      toastId = sonnerToast(message, toastOptions);
  }

  const id = toastId.toString();

  const update = (newProps: ToasterToast) => {
    sonnerToast.dismiss(toastId);
    return toast(newProps);
  };

  const dismiss = () => sonnerToast.dismiss(toastId);

  return {
    id,
    dismiss,
    update,
  };
}

function useToast() {
  const [state] = React.useState<State>({ toasts: [] });

  return {
    ...state,
    toast,
    dismiss: (toastId?: string) => {
      if (toastId) {
        sonnerToast.dismiss(toastId);
      } else {
        sonnerToast.dismiss();
      }
    },
  };
}

interface ToastOnFailOptions {
  rethrow?: boolean;
}

function useToastOnFail() {
  return React.useCallback(
    (action: string, { rethrow = false }: ToastOnFailOptions = {}) =>
      (error: any) => {
        const err = error as Error;
        toast({
          title: `Unable to ${action}`,
          description: err.message ?? "Something went wrong",
          variant: "destructive",
          duration: 10000,
        });
        if (rethrow) {
          throw error;
        }
      },
    [],
  );
}

export { toast, useToast, useToastOnFail };
