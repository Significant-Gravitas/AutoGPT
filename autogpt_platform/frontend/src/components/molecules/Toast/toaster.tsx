"use client";

import { Toaster as SonnerToaster } from "sonner";
import { CheckCircle, XCircle, Warning, Info } from "@phosphor-icons/react";
import styles from "./styles.module.css";

export function Toaster() {
  return (
    <SonnerToaster
      position="bottom-center"
      richColors
      toastOptions={{
        classNames: {
          toast: styles.toastDefault,
          title: styles.toastTitle,
          description: styles.toastDescription,
          error: styles.toastError,
          success: styles.toastSuccess,
          warning: styles.toastWarning,
          info: styles.toastInfo,
        },
      }}
      icons={{
        success: <CheckCircle className="h-4 w-4" color="#fff" weight="fill" />,
        error: <XCircle className="h-4 w-4" color="#fff" weight="fill" />,
        warning: <Warning className="h-4 w-4" color="#fff" weight="fill" />,
        info: <Info className="h-4 w-4" color="#fff" weight="fill" />,
      }}
    />
  );
}
