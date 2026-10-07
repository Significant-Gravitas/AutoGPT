import { cn } from "@/lib/utils";
import { cva, type VariantProps } from "class-variance-authority";
import { forwardRef } from "react";

const kbdVariants = cva(
  "inline-flex shrink-0 items-center justify-center rounded-md border border-border border-b-zinc-300 bg-card font-sans text-xs font-normal text-foreground shadow-xs",
  {
    variants: {
      size: {
        sm: "h-5 min-w-5 px-1",
        md: "h-6 min-w-6 px-1.5",
      },
    },
    defaultVariants: {
      size: "sm",
    },
  },
);

interface Props
  extends React.HTMLAttributes<HTMLElement>, VariantProps<typeof kbdVariants> {}

export const Kbd = forwardRef<HTMLElement, Props>(function Kbd(
  { className, size, ...props },
  ref,
) {
  return (
    <kbd
      ref={ref}
      className={cn(kbdVariants({ size }), "sentry-unmask", className)}
      {...props}
    />
  );
});
