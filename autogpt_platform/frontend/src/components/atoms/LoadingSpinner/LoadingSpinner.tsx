import { cn } from "@/lib/utils";
import React from "react";
import { Loading03Icon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";

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
} & Omit<React.ComponentPropsWithoutRef<typeof Icon>, "icon" | "size">;

export function LoadingSpinner(props: LoadingSpinnerProps) {
  const { size = "medium", className, cover = false, ...restProps } = props;

  const spinner = (
    <Icon
      icon={Loading03Icon}
      className={cn(
        "animate-spin text-inherit",
        sizeClassNameMap[size],
        className,
      )}
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
