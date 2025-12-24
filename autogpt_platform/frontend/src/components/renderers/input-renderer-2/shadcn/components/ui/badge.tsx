import { cva, type VariantProps } from 'class-variance-authority';
import { ComponentProps } from 'react';
import { Slot } from '@radix-ui/react-slot';

import { cn } from '../../lib/utils';

/**
 * Predefined badge variants using class-variance-authority
 * @see https://ui.shadcn.com/docs/components/badge
 */
const badgeVariants = cva(
  'inline-flex items-center justify-center rounded-md border px-2 py-0.5 text-xs font-medium w-fit whitespace-nowrap shrink-0 [&>svg]:size-3 gap-1 [&>svg]:pointer-events-none focus-visible:border-ring focus-visible:ring-ring/50 focus-visible:ring-[3px] aria-invalid:ring-destructive/20 dark:aria-invalid:ring-destructive/40 aria-invalid:border-destructive transition-[color,box-shadow] overflow-hidden',
  {
    variants: {
      variant: {
        default: 'border-transparent bg-primary text-primary-foreground [a&]:hover:bg-primary/90',
        secondary: 'border-transparent bg-secondary text-secondary-foreground [a&]:hover:bg-secondary/90',
        destructive:
          'border-transparent bg-destructive text-white [a&]:hover:bg-destructive/90 focus-visible:ring-destructive/20 dark:focus-visible:ring-destructive/40 dark:bg-destructive/60',
        outline: 'text-foreground [a&]:hover:bg-accent [a&]:hover:text-accent-foreground',
      },
    },
    defaultVariants: {
      variant: 'default',
    },
  },
);

/**
 * A badge component that displays short status descriptors
 *
 * @param asChild - If true, renders the badge as a child component (useful for custom components)
 * @param props - The props for the Badge component
 * @param props.className - Additional CSS classes to apply to the badge
 * @param props.variant - The style variant of the badge: 'default' | 'secondary' | 'destructive' | 'outline'
 * @returns A div element that displays as a badge
 */
function Badge({
  className,
  variant,
  asChild = false,
  ...props
}: ComponentProps<'span'> & VariantProps<typeof badgeVariants> & { asChild?: boolean }) {
  const Comp = asChild ? Slot : 'span';
  return <Comp data-slot='badge' className={cn(badgeVariants({ variant }), className)} {...props} />;
}

export { Badge, badgeVariants };
