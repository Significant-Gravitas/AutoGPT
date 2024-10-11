import * as React from "react";
import { Slot } from "@radix-ui/react-slot";
import { cva, type VariantProps } from "class-variance-authority";

import { cn } from "@/lib/utils";

const buttonVariants = cva(
  "inline-flex items-center justify-center whitespace-nowrap rounded-[80px] text-xl font-medium transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-neutral-950 disabled:pointer-events-none disabled:opacity-50 dark:focus-visible:ring-neutral-300 font-neue leading-9 tracking-tight",
  {
    variants: {
      variant: {
        default:
          "h-[70px] px-[26px] py-[7px] bg-white border border-black/50 text-[#272727]",
        destructive:
          "h-[70px] px-[26px] py-[7px] bg-red-500 text-neutral-50 border border-red-500/50 hover:bg-red-500/90",
        outline:
          "h-[70px] px-[26px] py-[7px] bg-white border border-black/50 text-[#272727] hover:bg-neutral-100",
        secondary:
          "h-[70px] px-[26px] py-[7px] bg-neutral-100 text-[#272727] border border-neutral-200 hover:bg-neutral-100/80",
        ghost:
          "h-[70px] px-[26px] py-[7px] hover:bg-neutral-100 text-[#272727]",
        link: "text-[#272727] underline-offset-4 hover:underline",
      },
      size: {
        default: "h-[70px] px-[26px] py-[7px]",
        sm: "h-[50px] px-[20px] py-[5px] text-sm",
        lg: "h-[90px] px-[32px] py-[9px] text-2xl",
        primary: "md:h-[70px] md:w-[176px] h-[50px] w-[112px]",
        icon: "h-[70px] w-[70px]",
      },
    },
    defaultVariants: {
      variant: "default",
      size: "default",
    },
  },
);

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {
  asChild?: boolean;
}

const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant, size, asChild = false, ...props }, ref) => {
    const Comp = asChild ? Slot : "button";
    return (
      <Comp
        className={cn(buttonVariants({ variant, size, className }))}
        ref={ref}
        {...props}
      />
    );
  },
);
Button.displayName = "Button";

export { Button, buttonVariants };
