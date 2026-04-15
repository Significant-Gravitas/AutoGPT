"use client";

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
        success: null,
        error: null,
        warning: null,
        info: null,
      }}
    />
  );
}
