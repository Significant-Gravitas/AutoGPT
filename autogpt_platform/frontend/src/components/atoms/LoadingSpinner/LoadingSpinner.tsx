import { cn } from "@/lib/utils";
import React from "react";
import { Loading03Icon } from "@hugeicons/core-free-icons";
import type { IconSvgElement } from "@hugeicons/react";
import { Icon } from "@/components/atoms/Icon/Icon";

const sizeClassNameMap = {
  small: "h-4 w-4",
  medium: "h-6 w-6",
  large: "h-10 w-10",
} as const;

type SpinnerSize = keyof typeof sizeClassNameMap;

// The open-ring spinner the old shadcn `loading` primitive drew (lucide's
// loader-circle), kept so its callers render exactly as before.
const CIRCLE_ICON: IconSvgElement = [
  [
    "path",
    {
      d: "M21 12a9 9 0 1 1-6.219-8.56",
      strokeLinecap: "round",
      strokeLinejoin: "round",
      key: "0",
    },
  ],
];

const iconMap = {
  default: Loading03Icon,
  circle: CIRCLE_ICON,
} as const;

type LoadingSpinnerProps = {
  size?: SpinnerSize;
  variant?: keyof typeof iconMap;
  className?: string;
  cover?: boolean;
} & Omit<React.ComponentPropsWithoutRef<typeof Icon>, "icon" | "size">;

export function LoadingSpinner(props: LoadingSpinnerProps) {
  const {
    size = "medium",
    variant = "default",
    className,
    cover = false,
    ...restProps
  } = props;

  const spinner = (
    <Icon
      data-slot="loading-spinner"
      icon={iconMap[variant]}
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
      <div
        data-slot="loading-spinner-cover"
        className="fixed inset-0 z-50 flex items-center justify-center"
      >
        {spinner}
      </div>
    );
  }

  return spinner;
}
