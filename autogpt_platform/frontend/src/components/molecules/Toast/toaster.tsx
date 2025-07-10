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
        success: <CheckCircle className="h-5 w-5" color="#fff" />,
        error: <XCircle className="h-5 w-5" color="#fff" />,
        warning: <Warning className="h-5 w-5" color="#fff" />,
        info: <Info className="h-5 w-5" color="#fff" />,
      }}
    />
  );
}
