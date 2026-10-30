import { cn } from "@/lib/utils";
import { forwardRef } from "react";

export interface BareInputProps extends React.InputHTMLAttributes<HTMLInputElement> {
  bare: true;
}

// The look of the deleted components/ui/input (SECRT-3113): just the <input>,
// no label, wrapper or error line. Every prop and handler reaches the element
// unchanged (no IME keydown guard, no value filtering), so it behaves exactly
// as that element did; a consumer's onKeyDown must use isKey itself.
export const BareInput = forwardRef<HTMLInputElement, BareInputProps>(
  function BareInput({ bare: _bare, className, type, ...props }, ref) {
    return (
      <input
        data-slot="input-control"
        type={type}
        className={cn(
          "flex h-9 w-full rounded-md border border-border-base bg-transparent px-3 py-1 text-body-lg leading-6 shadow-sm transition-colors file:border-0 file:bg-transparent file:text-body file:leading-5 file:font-medium file:text-foreground-base placeholder:text-placeholder focus-visible:ring-1 focus-visible:ring-focus focus-visible:outline-none disabled:cursor-not-allowed disabled:opacity-50 md:text-body md:leading-5",
          className,
        )}
        ref={ref}
        {...props}
      />
    );
  },
);

export interface BareTextareaProps extends React.TextareaHTMLAttributes<HTMLTextAreaElement> {
  bare: true;
  type: "textarea";
}

// The look of the deleted components/ui/textarea (SECRT-3113), on the same
// terms as BareInput above.
export const BareTextarea = forwardRef<HTMLTextAreaElement, BareTextareaProps>(
  function BareTextarea(
    { bare: _bare, type: _type, className, ...props },
    ref,
  ) {
    return (
      <textarea
        data-slot="input-control"
        className={cn(
          "flex min-h-[60px] w-full rounded-md border border-border-base bg-transparent px-3 py-2 text-body-lg leading-6 shadow-sm placeholder:text-placeholder focus-visible:ring-1 focus-visible:ring-focus focus-visible:outline-none disabled:cursor-not-allowed disabled:opacity-50 md:text-body md:leading-5",
          className,
        )}
        ref={ref}
        {...props}
      />
    );
  },
);
