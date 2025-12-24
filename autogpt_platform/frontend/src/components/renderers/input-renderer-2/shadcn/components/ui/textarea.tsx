import { TextareaHTMLAttributes, ComponentProps } from 'react';

import { cn } from '../../lib/utils';

/**
 * Props for the Textarea component
 * @extends TextareaHTMLAttributes<HTMLTextAreaElement> - HTML textarea element attributes
 */
export type TextareaProps = TextareaHTMLAttributes<HTMLTextAreaElement>;

/**
 * A textarea component with styling and focus states
 *
 * @param props - The props for the Textarea component
 * @param props.className - Additional CSS classes to apply to the textarea
 * @param ref - The forwarded ref for the textarea element
 * @returns A styled textarea element
 */
function Textarea({ className, ...props }: ComponentProps<'textarea'>) {
  return (
    <textarea
      data-slot='textarea'
      className={cn(
        'border-input placeholder:text-muted-foreground focus-visible:border-ring focus-visible:ring-ring/50 aria-invalid:ring-destructive/20 dark:aria-invalid:ring-destructive/40 aria-invalid:border-destructive dark:bg-input/30 flex field-sizing-content min-h-16 w-full rounded-md border bg-transparent px-3 py-2 text-base shadow-xs transition-[color,box-shadow] outline-none focus-visible:ring-[3px] disabled:cursor-not-allowed disabled:opacity-50 md:text-sm',
        className,
      )}
      {...props}
    />
  );
}

export { Textarea };
