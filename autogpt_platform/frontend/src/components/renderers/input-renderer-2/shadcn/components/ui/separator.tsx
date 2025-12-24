'use client';

import { Root } from '@radix-ui/react-separator';
import { ComponentProps } from 'react';

import { cn } from '../../lib/utils';

/**
 * A separator component for visually dividing content
 *
 * @param props - The props for the Separator component
 * @param props.className - Additional CSS classes to apply to the separator
 * @param props.orientation - The orientation of the separator ('horizontal' | 'vertical')
 * @param props.decorative - Whether the separator is decorative or semantic
 * @param ref - The forwarded ref for the separator element
 * @returns A styled separator element
 */
function Separator({
  className,
  orientation = 'horizontal',
  decorative = true,
  ...props
}: ComponentProps<typeof Root>) {
  return (
    <Root
      data-slot='separator'
      decorative={decorative}
      orientation={orientation}
      className={cn(
        'bg-border shrink-0 data-[orientation=horizontal]:h-px data-[orientation=horizontal]:w-full data-[orientation=vertical]:h-full data-[orientation=vertical]:w-px',
        className,
      )}
      {...props}
    />
  );
}

export { Separator };
