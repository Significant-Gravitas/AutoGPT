"use client";

import { Search01Icon } from "@hugeicons/core-free-icons";
import type { ChangeEvent } from "react";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Input } from "@/components/atoms/Input/Input";
import { cn } from "@/lib/utils";

interface Props {
  id: string;
  label: string;
  value: string;
  onChange: (next: string) => void;
  className?: string;
}

/** The team pages' search: the Files page's rounded Input with a leading
 *  search icon. */
export function ToolbarSearch({
  id,
  label,
  value,
  onChange,
  className,
}: Props) {
  function handleChange(event: ChangeEvent<HTMLInputElement>) {
    onChange(event.target.value);
  }

  return (
    <div
      className={cn(
        "relative flex w-full items-center max-sm:order-first sm:w-64",
        className,
      )}
    >
      <Icon
        icon={Search01Icon}
        size={16}
        className="pointer-events-none absolute left-4 top-1/2 z-20 -translate-y-1/2 text-zinc-500"
      />
      <Input
        id={id}
        label={label}
        hideLabel
        type="text"
        size="small"
        value={value}
        onChange={handleChange}
        placeholder={label}
        className="rounded-full pl-10"
        wrapperClassName="!mb-0 w-full"
      />
    </div>
  );
}
