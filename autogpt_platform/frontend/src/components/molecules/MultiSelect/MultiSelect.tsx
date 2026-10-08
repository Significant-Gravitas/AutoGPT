"use client";

import { Combobox as ComboboxPrimitive } from "@base-ui/react";
import { Cancel01Icon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Button } from "@/components/ui/button";
import {
  Combobox,
  ComboboxChip,
  ComboboxChips,
  ComboboxChipsInput,
  ComboboxContent,
  ComboboxEmpty,
  ComboboxItem,
  ComboboxList,
  useComboboxAnchor,
} from "@/components/ui/combobox";
import { cn } from "@/lib/utils";

export interface MultiSelectOption {
  value: string;
  label: string;
  description?: string;
  disabled?: boolean;
}

interface Props {
  options: MultiSelectOption[];
  value: string[];
  onValueChange: (value: string[]) => void;
  placeholder?: string;
  emptyMessage?: string;
  disabled?: boolean;
  id?: string;
  name?: string;
  className?: string;
  "aria-label"?: string;
  "aria-labelledby"?: string;
  "aria-describedby"?: string;
}

export function MultiSelect({
  options,
  value,
  onValueChange,
  placeholder = "Select options...",
  emptyMessage = "No results found",
  disabled,
  id,
  name,
  className,
  "aria-label": ariaLabel,
  "aria-labelledby": ariaLabelledBy,
  "aria-describedby": ariaDescribedBy,
}: Props) {
  const anchor = useComboboxAnchor();
  const byValue = new Map(options.map((option) => [option.value, option]));

  function labelOf(optionValue: string) {
    return byValue.get(optionValue)?.label ?? optionValue;
  }

  return (
    <Combobox
      items={options.map((option) => option.value)}
      multiple
      value={value}
      onValueChange={(next) => onValueChange(next)}
      itemToStringLabel={labelOf}
      disabled={disabled}
      name={name}
    >
      <ComboboxChips ref={anchor} className={cn("w-full", className)}>
        {value.map((selected) => (
          <ComboboxChip key={selected} showRemove={false}>
            {labelOf(selected)}
            <ChipRemove label={labelOf(selected)} />
          </ComboboxChip>
        ))}
        <ComboboxChipsInput
          id={id}
          placeholder={value.length === 0 ? placeholder : undefined}
          aria-label={ariaLabel}
          aria-labelledby={ariaLabelledBy}
          aria-describedby={ariaDescribedBy}
        />
      </ComboboxChips>
      <ComboboxContent anchor={anchor}>
        <ComboboxEmpty>{emptyMessage}</ComboboxEmpty>
        <ComboboxList>
          {(optionValue: string) => {
            const option = byValue.get(optionValue);
            return (
              <ComboboxItem
                key={optionValue}
                value={optionValue}
                disabled={option?.disabled}
                title={option?.description}
              >
                {labelOf(optionValue)}
              </ComboboxItem>
            );
          }}
        </ComboboxList>
      </ComboboxContent>
    </Combobox>
  );
}

// Kobra's built-in chip remove button has no accessible name and takes no
// props for one, so the molecule renders its own with the same styling.
interface ChipRemoveProps {
  label: string;
}

function ChipRemove({ label }: ChipRemoveProps) {
  return (
    <ComboboxPrimitive.ChipRemove
      aria-label={`Remove ${label}`}
      render={<Button variant="ghost" size="icon-xs" />}
      className="isolation-auto -ms-1 size-[calc(--spacing(5.25))] rounded-full opacity-50 before:-start-2.5 before:[mask-image:linear-gradient(to_right,transparent,black_62%)] hover:opacity-100 rtl:before:[mask-image:linear-gradient(to_left,transparent,black_62%)]"
      data-slot="combobox-chip-remove"
    >
      <Icon icon={Cancel01Icon} className="pointer-events-none" />
    </ComboboxPrimitive.ChipRemove>
  );
}
