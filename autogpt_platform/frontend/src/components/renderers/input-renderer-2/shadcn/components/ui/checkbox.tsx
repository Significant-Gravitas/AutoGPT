'use client';

import { Indicator, Root } from '@radix-ui/react-checkbox';
import { CheckIcon } from '@radix-ui/react-icons';
import { ComponentProps } from 'react';

import { cn } from '../../lib/utils';

/**
 * A checkbox component built on top of Radix UI Checkbox primitive
 * Renders an interactive checkbox that can be either checked or unchecked
 * @see https://ui.shadcn.com/docs/components/checkbox
 *
 * @param props - Props extending Radix UI Checkbox primitive props
 * @param props.className - Additional CSS classes to apply to the checkbox
 * @param ref - Forward ref to access the underlying checkbox element
 */
function Checkbox({ className, ...props }: ComponentProps<typeof Root>) {
  return (
    <Root
      data-slot='checkbox'
      className={cn(
        'peer border-input dark:bg-input/30 data-[state=checked]:bg-primary data-[state=checked]:text-primary-foreground dark:data-[state=checked]:bg-primary data-[state=checked]:border-primary focus-visible:border-ring focus-visible:ring-ring/50 aria-invalid:ring-destructive/20 dark:aria-invalid:ring-destructive/40 aria-invalid:border-destructive size-4 shrink-0 rounded-[4px] border shadow-xs transition-shadow outline-none focus-visible:ring-[3px] disabled:cursor-not-allowed disabled:opacity-50',
        className,
      )}
      {...props}
    >
      <Indicator
        data-slot='checkbox-indicator'
        className='flex items-center justify-center text-current transition-none'
      >
        <CheckIcon className='size-3.5' />
      </Indicator>
    </Root>
  );
}

export { Checkbox };
