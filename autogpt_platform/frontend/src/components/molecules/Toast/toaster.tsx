"use client";

import { CheckCircle, Info, Warning, XCircle } from "@phosphor-icons/react";
import { Toaster as SonnerToaster } from "sonner";
import styles from "./styles.module.css";

export function Toaster() {
  return (
    <SonnerToaster
      position="bottom-center"
      richColors
      closeButton
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
      className="custom__toast"
      icons={{
        success: <CheckCircle className="h-4 w-4" color="#fff" weight="fill" />,
        error: <XCircle className="h-4 w-4" color="#fff" weight="fill" />,
        warning: <Warning className="h-4 w-4" color="#fff" weight="fill" />,
        info: <Info className="h-4 w-4" color="#fff" weight="fill" />,
      }}
    />
  );
}
