'use client';

import { Check, ChevronDown } from 'lucide-react';
import { FocusEvent, FocusEventHandler, ReactElement, useEffect, useRef, useState } from 'react';
import { cn } from '../../lib/utils';
import { Command, CommandGroup, CommandItem, CommandList } from './command';

/**
 * Represents an item in the fancy select dropdown
 */
export type FancySelectItem = {
  /** The value of the item */
  value: any;
  /** The display label for the item */
  label: string;
  /** The index position of the item */
  index: number;
  /** Whether the item is disabled */
  disabled?: boolean;
};

/**
 * Props interface for the FancySelect component
 */
interface FancySelectInterface {
  /** Array of items to display in the dropdown */
  items: FancySelectItem[] | undefined;
  /** Currently selected item value */
  selected: string;
  /** Callback function when value changes */
  onValueChange?: (value: any) => void;
  /** Whether the component should autofocus */
  autoFocus?: boolean;
  /** ID of the element that describes this select */
  ariaDescribedby?: string;
  /** Aria placeholder text */
  ariaPlaceholder?: string;
  /** Additional className for styling */
  className?: string;
  /** Whether the select is disabled */
  disabled?: boolean;
  /** Callback function for blur event */
  onBlur?: FocusEventHandler<HTMLDivElement> | undefined;
  /** Callback function for focus event */
  onFocus?: FocusEventHandler<HTMLDivElement> | undefined;
  /** Whether the field is required */
  required?: boolean;
  /** Placeholder text when no item is selected */
  placeholder?: string;
}

/**
 * A fancy select component that provides a styled dropdown with search functionality
 * @param props - The component props
 * @returns A React component that renders a searchable select dropdown
 */
export function FancySelect({
  items,
  selected,
  onValueChange,
  autoFocus = false,
  disabled = false,
  required = false,
  placeholder = 'Select...',
  ariaDescribedby,
  ariaPlaceholder,
  onFocus,
  onBlur,
  className,
}: Readonly<FancySelectInterface>): ReactElement {
  const [open, setOpen] = useState(false);
  const selectedItem = items?.find((item) => item.value === selected);
  const selectedRef = useRef<HTMLDivElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (open && selectedRef.current) {
      requestAnimationFrame(() => {
        selectedRef.current?.scrollIntoView({ block: 'nearest' });
      });
    }
  }, [open]);

  const handleBlur = (e: FocusEvent<HTMLDivElement>) => {
    if (containerRef.current && !containerRef.current.contains(e.relatedTarget as Node)) {
      setOpen(false);
    }
    onBlur?.(e);
  };

  return (
    <Command
      ref={containerRef}
      className={cn('overflow-visible bg-transparent', className)}
      autoFocus={autoFocus}
      aria-disabled={disabled}
      onBlur={handleBlur}
      onFocus={onFocus}
      aria-describedby={ariaDescribedby}
      aria-placeholder={ariaPlaceholder}
    >
      <div
        onClick={() => !disabled && setOpen(!open)}
        className={cn(
          'flex h-9 w-full items-center justify-between gap-2 whitespace-nowrap rounded-md border border-input bg-transparent px-3 py-2 text-sm shadow-sm ring-offset-background placeholder:text-muted-foreground focus:outline-none focus:ring-1 focus:ring-ring disabled:cursor-not-allowed disabled:opacity-50',
          !selectedItem && required && 'border-red-500',
          disabled && 'opacity-50 cursor-not-allowed',
          className,
        )}
      >
        <span className={cn('flex-1 line-clamp-1', !selectedItem && 'text-muted-foreground')}>
          {selectedItem?.label || placeholder}
        </span>
        <ChevronDown className='h-4 w-4 opacity-50' />
      </div>
      <div className='relative'>
        {open && items && items.length > 0 ? (
          <div
            style={{ top: '0.5rem' }}
            className='absolute w-full z-10 rounded-md border bg-popover text-popover-foreground shadow-md outline-none'
          >
            <CommandGroup className='h-full overflow-auto'>
              <CommandList>
                {items.map((item) => (
                  <CommandItem
                    ref={item.value === selected ? selectedRef : undefined}
                    key={item.value}
                    onMouseDown={(e) => {
                      e.preventDefault();
                      e.stopPropagation();
                    }}
                    onSelect={() => {
                      if (!item.disabled) {
                        onValueChange?.(item.value);
                        setOpen(false);
                      }
                    }}
                    className={cn(
                      'cursor-pointer relative flex items-center justify-between rounded-sm py-1.5 gap-2 rtl:flex-row-reverse',
                      item.value === selected && 'font-semibold',
                      item.disabled && 'opacity-50 cursor-not-allowed',
                    )}
                  >
                    <span>{item.label}</span>
                    <span className='flex h-3.5 w-3.5 items-center justify-center'>
                      {item.value === selected && <Check className='h-4 w-4' />}
                    </span>
                  </CommandItem>
                ))}
              </CommandList>
            </CommandGroup>
          </div>
        ) : null}
      </div>
    </Command>
  );
}
