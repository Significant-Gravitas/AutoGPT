import { cn } from "@/lib/utils";
import { CircleNotchIcon } from "@phosphor-icons/react/dist/ssr";
import React from "react";

const sizeClassNameMap = {
  small: "h-4 w-4",
  medium: "h-6 w-6",
  large: "h-10 w-10",
} as const;

type SpinnerSize = keyof typeof sizeClassNameMap;

type LoadingSpinnerProps = {
  size?: SpinnerSize;
  className?: string;
  cover?: boolean;
} & React.ComponentPropsWithoutRef<typeof CircleNotchIcon>;

export function LoadingSpinner(props: LoadingSpinnerProps) {
  const { size = "medium", className, cover = false, ...restProps } = props;

  const spinner = (
    <CircleNotchIcon
      className={cn(
        "animate-spin text-inherit",
        sizeClassNameMap[size],
        className,
      )}
      weight="bold"
      {...restProps}
    />
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
