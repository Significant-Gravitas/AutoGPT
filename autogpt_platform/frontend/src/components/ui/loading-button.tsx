import React from "react";
import { Button, ButtonProps } from "./button";
import { LoadingSpinner } from "./loading";
import { cn } from "@/lib/utils";

interface LoadingButtonProps extends ButtonProps {
  loading?: boolean;
  loadingText?: string;
  children: React.ReactNode;
}

export function LoadingButton({
  loading = false,
  loadingText,
  children,
  disabled,
  className,
  ...props
}: LoadingButtonProps) {
  return (
    <Button {...props} disabled={loading || disabled} className={cn(className)}>
      {loading ? (
        <>
          <LoadingSpinner className="mr-2 h-4 w-4" />
          {loadingText || "Loading..."}
        </>
      ) : (
        children
      )}
    </Button>
  );
}
