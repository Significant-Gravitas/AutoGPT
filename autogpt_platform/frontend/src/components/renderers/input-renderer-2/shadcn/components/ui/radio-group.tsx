'use client';

import { Root, Indicator, Item } from '@radix-ui/react-radio-group';
import { ComponentProps } from 'react';
import { CircleIcon } from 'lucide-react';

import { cn } from '../../lib/utils';

/**
 * A radio group component for selecting a single option from a list
 *
 * @param props - The props for the RadioGroup component
 * @param props.className - Additional CSS classes to apply to the radio group
 * @param ref - The forwarded ref for the radio group element
 * @returns A radio group container element
 */
function RadioGroup({ className, ...props }: ComponentProps<typeof Root>) {
  return <Root data-slot='radio-group' className={cn('grid gap-3', className)} {...props} />;
}

/**
 * An individual radio item within a RadioGroup
 *
 * @param props - The props for the RadioGroupItem component
 * @param props.className - Additional CSS classes to apply to the radio item
 * @param ref - The forwarded ref for the radio item element
 * @returns A styled radio input element
 */
function RadioGroupItem({ className, ...props }: ComponentProps<typeof Item>) {
  return (
    <Item
      data-slot='radio-group-item'
      className={cn(
        'border-input text-primary focus-visible:border-ring focus-visible:ring-ring/50 aria-invalid:ring-destructive/20 dark:aria-invalid:ring-destructive/40 aria-invalid:border-destructive dark:bg-input/30 aspect-square size-4 shrink-0 rounded-full border shadow-xs transition-[color,box-shadow] outline-none focus-visible:ring-[3px] disabled:cursor-not-allowed disabled:opacity-50',
        className,
      )}
      {...props}
    >
      <Indicator data-slot='radio-group-indicator' className='relative flex items-center justify-center'>
        <CircleIcon className='fill-primary absolute top-1/2 left-1/2 size-2 -translate-x-1/2 -translate-y-1/2' />
      </Indicator>
    </Item>
  );
}

export { RadioGroup, RadioGroupItem };
