"use client";

import { dismissToast, toast as showToast } from "@/components/ui/toast";
import * as React from "react";
import {
  PERSISTENT_LIFETIME,
  STATE_BY_VARIANT,
  toAction,
  toText,
  ToastVariant,
} from "./helpers";

export interface ToastProps {
  title?: React.ReactNode;
  description?: React.ReactNode;
  variant?: ToastVariant;
  /** Milliseconds on screen; Kobra picks a fitting lifetime when unset. */
  duration?: number;
  /** A button-like node: its text is the label, its `onClick` runs. */
  action?: React.ReactNode;
  dismissable?: boolean;
}

interface Toast extends ToastProps {
  id?: string;
}

let nextId = 0;
const issued = new Set<string>();

function toast({
  title,
  description,
  variant = "default",
  duration,
  action,
  dismissable = true,
  id = `toast-${++nextId}`,
}: Toast) {
  const heading = toText(title);
  const detail = toText(description);

  showToast({
    id,
    message: heading || detail,
    description: heading && detail ? detail : undefined,
    state: STATE_BY_VARIANT[variant],
    action: toAction(action),
    lifetime: dismissable ? duration : PERSISTENT_LIFETIME,
  });
  issued.add(id);

  return {
    id,
    dismiss: () => dismiss(id),
    update: (next: ToastProps) => toast({ ...next, id }),
  };
}

function dismiss(toastId?: string) {
  const ids = toastId ? [toastId] : [...issued];
  for (const id of ids) {
    dismissToast(id);
    issued.delete(id);
  }
}

function useToast() {
  return { toast, dismiss };
}

interface ToastOnFailOptions {
  rethrow?: boolean;
}

function useToastOnFail() {
  return React.useCallback(
    (action: string, { rethrow = false }: ToastOnFailOptions = {}) =>
      (error: unknown) => {
        const message =
          error instanceof Error ? error.message : "Something went wrong";
        toast({
          title: `Unable to ${action}`,
          description: message,
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
