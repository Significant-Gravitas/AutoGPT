"use client";

import * as React from "react";
import hotToast from "react-hot-toast";

import { XIcon } from "@/components/atoms/AGPTIcon/icons";
import { cn } from "@/lib/utils";

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

type Variant = NonNullable<ToastProps["variant"]>;

const VARIANT_BACKGROUND: Record<Variant, string> = {
  default: "bg-[#2c2c30]",
  info: "bg-[#2c2c30]",
  success: "bg-[#45b56e]",
  destructive: "bg-[#f26969]",
};

interface ToastContentProps {
  toastId: string;
  visible: boolean;
  title?: React.ReactNode;
  description?: React.ReactNode;
  variant: Variant;
  action?: React.ReactNode;
  dismissable: boolean;
}

function ToastContent({
  toastId,
  visible,
  title,
  description,
  variant,
  action,
  dismissable,
}: ToastContentProps) {
  return (
    <div
      role="status"
      className={cn(
        "pointer-events-auto flex w-full min-w-[16rem] max-w-md items-start gap-3 rounded-lg px-4 py-3 shadow-lg transition-opacity duration-150",
        VARIANT_BACKGROUND[variant],
        visible ? "opacity-100" : "opacity-0",
      )}
    >
      <div className="min-w-0 flex-1">
        {title && (
          <p className="text-sm font-medium leading-[1.375rem] text-[#fefefe]">
            {title}
          </p>
        )}
        {description && (
          <p className="text-xs leading-5 text-[#f4f4f5]">{description}</p>
        )}
      </div>
      {action}
      {dismissable && (
        <button
          type="button"
          aria-label="Dismiss"
          onClick={() => hotToast.dismiss(toastId)}
          className="shrink-0 text-[#fefefe]/70 transition-colors hover:text-[#fefefe]"
        >
          <XIcon className="size-4" weight="bold" />
        </button>
      )}
    </div>
  );
}

function toast({
  title,
  description,
  variant = "default",
  duration = 5000,
  action,
  dismissable = true,
  ..._props
}: Toast) {
  const id = hotToast.custom(
    (t) => (
      <ToastContent
        toastId={t.id}
        visible={t.visible}
        title={title}
        description={description}
        variant={variant}
        action={action}
        dismissable={dismissable}
      />
    ),
    { duration: dismissable ? duration : Infinity },
  );

  const update = (newProps: ToasterToast) => {
    hotToast.dismiss(id);
    return toast(newProps);
  };

  const dismiss = () => hotToast.dismiss(id);

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
        hotToast.dismiss(toastId);
      } else {
        hotToast.dismiss();
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
      (error: unknown) => {
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
