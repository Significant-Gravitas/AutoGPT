import { cn } from "@/lib/utils";
import { forwardRef } from "react";

// `bare`: no box at all, a search field set into a bar that draws its own
// (the builder's block and workflow search).
const VARIANTS = {
  bare: [
    "m-0 border-none p-0 font-sans text-body-lg leading-6 font-normal text-foreground-emphasis shadow-none outline-none",
    "placeholder:text-foreground-subtle focus:shadow-none focus:ring-0 focus:outline-none",
  ],
} as const;

export type BaseInputProps = React.InputHTMLAttributes<HTMLInputElement>;

interface Props extends BaseInputProps {
  variant?: keyof typeof VARIANTS;
}

// The bare <input> under the Input atom, for fields that bring their own label
// and layout.
export const BaseInput = forwardRef<HTMLInputElement, Props>(function BaseInput(
  { className, type, variant, ...props },
  ref,
) {
  return (
    <input
      type={type}
      className={cn(
        "agpt-border-input agpt-shadow-input flex h-9 w-full rounded-md border bg-transparent px-3 py-1 text-body leading-5 transition-colors file:border-0 file:bg-transparent file:text-body file:leading-5 file:font-medium placeholder:text-placeholder disabled:cursor-not-allowed disabled:opacity-50",
        type == "file" ? "pt-1.5 pb-0.5" : "",
        variant && VARIANTS[variant],
        className,
      )}
      ref={ref}
      {...props}
    />
  );
});
