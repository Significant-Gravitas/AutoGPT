'use client';

import { Command as CommandPrimitive } from 'cmdk';
import isEqual from 'lodash/isEqual';
import { X } from 'lucide-react';
import {
  FocusEvent,
  FocusEventHandler,
  KeyboardEvent,
  ReactElement,
  useCallback,
  useMemo,
  useRef,
  useState,
} from 'react';

import { cn } from '../../lib/utils';
import { Badge } from './badge';
import { Command, CommandGroup, CommandItem, CommandList } from './command';

/**
 * Represents an item in the fancy multi-select dropdown
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
 * Props interface for the FancyMultiSelect component
 */
interface FancyMultiSelectProps {
  /** Whether multiple items can be selected */
  multiple: boolean;
  /** Array of items to display in the dropdown */
  items?: FancySelectItem[];
  /** Array of selected item values */
  selected: string[];
  /** Callback function when value changes */
  onValueChange?: (value: number[]) => void;
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
  onBlur?: FocusEventHandler<HTMLDivElement>;
  /** Callback function for focus event */
  onFocus?: FocusEventHandler<HTMLDivElement>;
  /** Unique identifier for the component */
  id: string;
}

/**
 * A fancy multi-select component that allows users to select multiple items from a dropdown
 * @param props - The component props
 * @returns A React component that renders a searchable multi-select dropdown with tags
 */
export function FancyMultiSelect({
  multiple,
  items = [],
  selected,
  onValueChange,
  autoFocus = false,
  disabled = false,
  ariaDescribedby,
  ariaPlaceholder,
  onFocus,
  onBlur,
  className,
  id,
}: Readonly<FancyMultiSelectProps>): ReactElement {
  const inputRef = useRef<HTMLInputElement>(null);
  const [open, setOpen] = useState(false);
  const [inputValue, setInputValue] = useState('');

  const selectedItems = useMemo(
    () => items.filter((item) => selected.some((selectedValue) => isEqual(item.value, selectedValue))),
    [items, selected],
  );

  const selectables = useMemo(
    () => items.filter((framework) => !selectedItems.some((item) => isEqual(item.value, framework.value))),
    [items, selectedItems],
  );

  const handleUnselect = useCallback(
    (framework: FancySelectItem) => {
      if (disabled) {
        return;
      }
      const newSelected = selectedItems.filter((s) => !isEqual(s.value, framework.value));
      onValueChange?.(newSelected.map((item) => item.index));
    },
    [selectedItems, onValueChange, disabled],
  );

  const handleKeyDown = useCallback(
    (e: KeyboardEvent<HTMLDivElement>) => {
      if (disabled || !inputRef.current || inputRef.current.value !== '') {
        return;
      }

      if (e.key === 'Delete' || e.key === 'Backspace') {
        const newSelected = selectedItems.slice(0, -1);
        onValueChange?.(newSelected.map((item) => item.index));
      } else if (e.key === 'Escape') {
        inputRef.current.blur();
      }
    },
    [selectedItems, onValueChange, disabled],
  );

  const handleSelect = useCallback(
    (item: FancySelectItem) => {
      if (disabled) {
        return;
      }
      setInputValue('');
      const newSelected = multiple ? [...selectedItems, item] : [item];
      onValueChange?.(newSelected.map((item) => item.index));
    },
    [multiple, selectedItems, onValueChange, disabled],
  );

  const handleFocus = useCallback(
    (e: FocusEvent<HTMLDivElement>) => {
      if (!disabled) {
        setOpen(true);
      }
      onFocus?.(e);
    },
    [disabled, onFocus],
  );

  return (
    <Command
      onKeyDown={handleKeyDown}
      className={cn('overflow-visible bg-transparent', className)}
      autoFocus={autoFocus}
      aria-disabled={disabled}
      onBlur={onBlur}
      onFocus={handleFocus}
      aria-describedby={ariaDescribedby}
      aria-placeholder={ariaPlaceholder}
    >
      <div
        className={cn(
          'group border border-input px-3 py-2 text-sm ring-offset-background rounded-md focus-within:ring-1 focus-within:ring-ring focus-within:ring-offset-1',
          disabled && 'opacity-50 cursor-not-allowed',
        )}
      >
        <div className='flex gap-1 flex-wrap'>
          {selectedItems.map((item) => (
            <Badge key={item.value} variant='secondary'>
              {item.label}
              <button
                type='button'
                className='rtl:mr-1 ltr:ml-1 ring-offset-background rounded-full outline-none focus:ring-1 focus:ring-ring focus:ring-offset-1'
                onKeyDown={(e) => e.key === 'Enter' && !disabled && handleUnselect(item)}
                onMouseDown={(e) => {
                  e.preventDefault();
                  e.stopPropagation();
                }}
                onClick={() => handleUnselect(item)}
                disabled={disabled}
              >
                <X
                  className={cn(
                    'h-3 w-3 text-muted-foreground hover:text-foreground',
                    disabled && 'pointer-events-none',
                  )}
                />
              </button>
            </Badge>
          ))}
          <CommandPrimitive.Input
            ref={inputRef}
            value={inputValue}
            onValueChange={setInputValue}
            onBlur={() => setOpen(false)}
            onFocus={() => !disabled && setOpen(true)}
            placeholder='Select ...'
            className='rtl:mr-2 ltr:ml-2 bg-transparent outline-none placeholder:text-muted-foreground flex-1'
            disabled={disabled}
            aria-controls={`command-item-input-${id}`}
            aria-labelledby={`command-item-input-${id}`}
            id={`command-item-input-${id}`}
          />
        </div>
      </div>
      {open && !disabled && selectables.length > 0 && (
        <div className='relative mt-2'>
          <div className='absolute w-full z-10 top-0 rounded-md border bg-popover text-popover-foreground shadow-md outline-none animate-in'>
            <CommandGroup className='h-full overflow-auto'>
              <CommandList>
                {selectables.map((item) => (
                  <CommandItem
                    disabled={item.disabled}
                    key={`${item.value}-command-item`}
                    onMouseDown={(e) => {
                      e.preventDefault();
                      e.stopPropagation();
                    }}
                    aria-controls={`${item.value}-command-item`}
                    aria-labelledby={`${item.value}-command-item`}
                    id={`${item.value}-command-item`}
                    onSelect={() => handleSelect(item)}
                    className='cursor-pointer'
                  >
                    {item.label}
                  </CommandItem>
                ))}
              </CommandList>
            </CommandGroup>
          </div>
        </div>
      )}
    </Command>
  );
}
