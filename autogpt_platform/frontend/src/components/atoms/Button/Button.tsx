import { cn } from "@/lib/utils";
import { CircleNotchIcon } from "@phosphor-icons/react/dist/ssr";
import { cva, type VariantProps } from "class-variance-authority";
import React from "react";

// Extended button variants based on our design system
const extendedButtonVariants = cva(
  "inline-flex items-center justify-center whitespace-nowrap font-medium transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-neutral-950 disabled:pointer-events-none disabled:opacity-50 font-['Geist'] leading-snug border",
  {
    variants: {
      variant: {
        primary:
          "bg-zinc-800 border-zinc-800 text-white hover:bg-zinc-900 hover:border-zinc-900 rounded-full disabled:text-white disabled:bg-zinc-200 disabled:border-zinc-200 disabled:opacity-1",
        secondary:
          "bg-zinc-100 border-zinc-100 text-black hover:bg-zinc-300 hover:border-zinc-300 rounded-full disabled:text-zinc-300 disabled:bg-zinc-50 disabled:border-zinc-50 disabled:opacity-1",
        destructive:
          "bg-red-500 border-red-500 text-white hover:bg-red-600 hover:border-red-600 rounded-full disabled:text-white disabled:bg-zinc-200 disabled:border-zinc-200 disabled:opacity-1",
        outline:
          "bg-transparent border-zinc-700 text-black hover:bg-zinc-100 hover:border-zinc-700 rounded-full disabled:border-zinc-200 disabled:text-zinc-200 disabled:opacity-1",
        ghost:
          "bg-transparent border-transparent text-black hover:bg-zinc-50 hover:border-zinc-50 rounded-full disabled:text-zinc-200 disabled:opacity-1",
        link: "bg-transparent border-transparent text-black hover:underline rounded-none p-0 h-auto min-w-auto disabled:opacity-1",
        icon: "bg-white text-black border border-zinc-600 hover:bg-zinc-100 rounded-[96px] disabled:opacity-1",
      },
      size: {
        small: "px-3 py-2 text-sm gap-1.5 h-[2.25rem]",
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
  const isDisabled = disabled;

  return (
    <button
      className={cn(
        extendedButtonVariants({ variant, size, className }),
        loading && "pointer-events-none",
      )}
      disabled={isDisabled}
      {...props}
    >
      {loading && (
        <CircleNotchIcon className="h-4 w-4 animate-spin" weight="bold" />
      )}
      {!loading && leftIcon}
      {children}
      {!loading && rightIcon}
    </button>
  );
}

Button.displayName = "Button";

export { Button, extendedButtonVariants };
