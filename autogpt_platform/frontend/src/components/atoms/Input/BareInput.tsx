import { cn } from "@/lib/utils";
import { forwardRef } from "react";

export interface BareInputProps
  extends React.InputHTMLAttributes<HTMLInputElement> {
  bare: true;
}

// The look of the deleted components/ui/input (SECRT-3113): just the <input>,
// no label, wrapper or error line. Every prop and handler reaches the element
// unchanged (no IME keydown guard, no value filtering), so it behaves exactly
// as that element did; a consumer's onKeyDown must use isKey itself.
// TODO(tokens): no role holds neutral-200 #E5E5E5 (border is #E4E4E7) or
// neutral-500 #737373. neutral-950 #0A0A0A is the focus role, a ring colour,
// not a text colour; the ring stays raw too, because focus has no .dark value
// and this ring is neutral-300 in dark mode (SECRT-3154). The deleted file's
// dark: classes are carried verbatim.
export const BareInput = forwardRef<HTMLInputElement, BareInputProps>(
  function BareInput({ bare: _bare, className, type, ...props }, ref) {
    return (
      <input
        data-slot="input-control"
        type={type}
        className={cn(
          "flex h-9 w-full rounded-md border border-neutral-200 bg-transparent px-3 py-1 text-body-lg leading-6 shadow-sm transition-colors file:border-0 file:bg-transparent file:text-body file:font-medium file:leading-5 file:text-neutral-950 placeholder:text-neutral-500 focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-neutral-950 disabled:cursor-not-allowed disabled:opacity-50 dark:border-neutral-800 dark:file:text-neutral-50 dark:placeholder:text-neutral-400 dark:focus-visible:ring-neutral-300 md:text-body md:leading-5",
          className,
        )}
        ref={ref}
        {...props}
      />
    );
  },
);

export interface BareTextareaProps
  extends React.TextareaHTMLAttributes<HTMLTextAreaElement> {
  bare: true;
  type: "textarea";
}

// The look of the deleted components/ui/textarea (SECRT-3113), on the same
// terms as BareInput above.
// TODO(tokens): no role holds neutral-200 #E5E5E5 (border is #E4E4E7) or
// neutral-500 #737373; the ring stays raw neutral-950 because focus has no
// .dark value and this ring is neutral-300 in dark mode (SECRT-3154). The
// deleted file's dark: classes are carried verbatim.
export const BareTextarea = forwardRef<HTMLTextAreaElement, BareTextareaProps>(
  function BareTextarea(
    { bare: _bare, type: _type, className, ...props },
    ref,
  ) {
    return (
      <textarea
        data-slot="input-control"
        className={cn(
          "flex min-h-[60px] w-full rounded-md border border-neutral-200 bg-transparent px-3 py-2 text-body-lg leading-6 shadow-sm placeholder:text-neutral-500 focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-neutral-950 disabled:cursor-not-allowed disabled:opacity-50 dark:border-neutral-800 dark:placeholder:text-neutral-400 dark:focus-visible:ring-neutral-300 md:text-body md:leading-5",
          className,
        )}
        ref={ref}
        {...props}
      />
    );
  },
);
