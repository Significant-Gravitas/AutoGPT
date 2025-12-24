import { cva, type VariantProps } from 'class-variance-authority';
import { ComponentProps } from 'react';

import { cn } from '../../lib/utils';

const alertVariants = cva(
  'relative w-full rounded-lg border px-4 py-3 text-sm grid has-[>svg]:grid-cols-[calc(var(--spacing)*4)_1fr] grid-cols-[0_1fr] has-[>svg]:gap-x-3 gap-y-0.5 items-start [&>svg]:size-4 [&>svg]:translate-y-0.5 [&>svg]:text-current',
  {
    variants: {
      variant: {
        default: 'bg-card text-card-foreground',
        destructive:
          'text-destructive bg-card [&>svg]:text-current *:data-[slot=alert-description]:text-destructive/90',
      },
    },
    defaultVariants: {
      variant: 'default',
    },
  },
);

/** A component that displays a brief, important message in a way that attracts the user's attention without interrupting their task.
 *
 * @param props - Component props
 * @param props.variant - 'default' | 'destructive' - Style variant of the alert
 * @param props.className - Additional CSS classes
 * @returns A div element that serves as an alert component
 */
function Alert({ className, variant, ...props }: ComponentProps<'div'> & VariantProps<typeof alertVariants>) {
  return <div data-slot='alert' role='alert' className={cn(alertVariants({ variant }), className)} {...props} />;
}

/** Represents the title content of an Alert component.
 *
 * @param props - Component props
 * @param props.className - Additional CSS classes
 * @returns A heading element for the alert title
 */
function AlertTitle({ className, ...props }: ComponentProps<'div'>) {
  return (
    <div
      data-slot='alert-title'
      className={cn('col-start-2 line-clamp-1 min-h-4 font-medium tracking-tight', className)}
      {...props}
    />
  );
}

/** Represents the description content of an Alert component.
 *
 * @param props - Component props
 * @param props.className - Additional CSS classes
 * @returns A div element containing the alert description
 */
function AlertDescription({ className, ...props }: ComponentProps<'div'>) {
  return (
    <div
      data-slot='alert-description'
      className={cn(
        'text-muted-foreground col-start-2 grid justify-items-start gap-1 text-sm [&_p]:leading-relaxed',
        className,
      )}
      {...props}
    />
  );
}

export { Alert, AlertDescription, AlertTitle };
