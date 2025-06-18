import { cn } from "@/lib/utils";
import { CircleNotchIcon } from "@phosphor-icons/react/dist/ssr";
import { cva, type VariantProps } from "class-variance-authority";
import React from "react";

// Extended button variants based on our design system
const extendedButtonVariants = cva(
  "inline-flex items-center justify-center whitespace-nowrap font-medium transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-neutral-950 disabled:pointer-events-none disabled:opacity-50 font-['Geist'] leading-snug",
  {
    variants: {
      variant: {
        primary: "bg-zinc-700 text-white hover:bg-zinc-800 rounded-[43px]",
        secondary:
          "bg-zinc-200 text-neutral-900 hover:bg-zinc-400 rounded-[43px]",
        destructive: "bg-red-500 text-white hover:bg-red-600 rounded-[43px]",
        outline:
          "bg-white text-neutral-900 border border-zinc-600 hover:bg-zinc-200 rounded-[43px]",
        ghost:
          "bg-transparent text-neutral-900 hover:bg-zinc-100 rounded-[43px]",
        loading: "bg-zinc-500 text-white rounded-[43px] cursor-not-allowed",
        link: "bg-transparent text-neutral-900 hover:underline rounded-none p-0 h-auto min-w-0",
        icon: "bg-white text-neutral-900 border border-zinc-600 hover:bg-zinc-100 rounded-[96px]",
      },
      size: {
        small: "min-w-16 px-3 py-1.5 text-sm gap-1.5",
        large: "min-w-20 px-4 py-3 text-sm gap-2",
        icon: "p-3",
      },
    },
    defaultVariants: {
      variant: "primary",
      size: "large",
    },
  },
);

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof extendedButtonVariants> {
  loading?: boolean;
  leftIcon?: React.ReactNode;
  rightIcon?: React.ReactNode;
  asChild?: boolean;
}

function Button({
  className,
  variant,
  size,
  loading = false,
  leftIcon,
  rightIcon,
  children,
  disabled,
  ...props
}: ButtonProps) {
  const isDisabled = disabled || loading;
  const finalVariant = loading ? "loading" : variant;

  return (
    <button
      className={cn(
        extendedButtonVariants({ variant: finalVariant, size, className }),
      )}
      disabled={isDisabled}
      {...props}
    >
      {loading && (
        <CircleNotchIcon className="h-4 w-4 animate-spin" weight="thin" />
      )}
      {!loading && leftIcon}
      {children}
      {!loading && rightIcon}
    </button>
  );
}

Button.displayName = "Button";

export { Button, extendedButtonVariants };
