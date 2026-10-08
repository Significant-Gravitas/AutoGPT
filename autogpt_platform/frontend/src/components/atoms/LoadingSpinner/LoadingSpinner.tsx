import { Spinner } from "@/components/ui/spinner";
import { cn } from "@/lib/utils";
import React from "react";

const sizeClassNameMap = {
  small: "size-4",
  medium: "size-6",
  large: "size-10",
} as const;

type SpinnerSize = keyof typeof sizeClassNameMap;

type LoadingSpinnerProps = {
  size?: SpinnerSize;
  className?: string;
  cover?: boolean;
} & Omit<React.ComponentProps<typeof Spinner>, "className">;

export function LoadingSpinner(props: LoadingSpinnerProps) {
  const { size = "medium", className, cover = false, ...restProps } = props;

  const spinner = (
    <Spinner className={cn(sizeClassNameMap[size], className)} {...restProps} />
  );

  if (cover) {
    return (
      <div className="fixed inset-0 z-50 flex items-center justify-center">
        {spinner}
      </div>
    );
  }

  return spinner;
}
