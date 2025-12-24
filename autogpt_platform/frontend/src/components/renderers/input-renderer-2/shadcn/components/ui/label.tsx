'use client';

import { Root } from '@radix-ui/react-label';
import { ComponentProps } from 'react';

import { cn } from '../../lib/utils';

/**
 * A label component with styling variants
 *
 * @param props - The props for the Label component
 * @param props.className - Additional CSS classes to apply to the label
 * @param ref - The forwarded ref for the label element
 * @returns A styled label element
 */
function Label({ className, ...props }: ComponentProps<typeof Root>) {
  return (
    <Root
      data-slot='label'
      className={cn(
        'flex items-center gap-2 text-sm leading-none font-medium select-none group-data-[disabled=true]:pointer-events-none group-data-[disabled=true]:opacity-50 peer-disabled:cursor-not-allowed peer-disabled:opacity-50',
        className,
      )}
      {...props}
    />
  );
}

export { Label };
