"use client";

import { Icon } from "@/components/atoms/Icon/Icon";
import { Input } from "@/components/atoms/Input/Input";
import {
  Cancel01Icon,
  Loading03Icon,
  Search01Icon,
} from "@hugeicons/core-free-icons";
import type { KitSearchScope } from "./helpers";

interface Props {
  scope: KitSearchScope;
  label: string;
  placeholder: string;
  value: string;
  isSearching: boolean;
  onChange: (value: string) => void;
}

export function KitSearchField({
  scope,
  label,
  placeholder,
  value,
  isSearching,
  onChange,
}: Props) {
  return (
    // The Input atom wraps itself in two positioned divs that follow these
    // adornments in the DOM, so without z-10 the opaque field paints over them.
    <div className="relative w-full max-w-[42rem]">
      <Icon
        icon={Search01Icon}
        size={16}
        aria-hidden
        className="pointer-events-none absolute left-3.5 top-1/2 z-10 -translate-y-1/2 text-muted-foreground"
      />
      <Input
        id={`raise-${scope}-search`}
        label={label}
        hideLabel
        size="small"
        value={value}
        onChange={(event) => onChange(event.target.value)}
        placeholder={placeholder}
        className="pl-10 pr-11"
        wrapperClassName="mb-0 w-full [&_input]:h-[2.625rem] [&_input]:py-3"
      />
      {/* The spinner replaces the clear button rather than sitting beside it:
          while a query is in flight there is nothing settled to clear yet. */}
      {isSearching ? (
        <Icon
          icon={Loading03Icon}
          size={16}
          aria-hidden
          className="absolute right-3.5 top-1/2 z-10 -translate-y-1/2 animate-spin text-muted-foreground motion-reduce:animate-none"
        />
      ) : value ? (
        <button
          type="button"
          onClick={() => onChange("")}
          aria-label="Clear search"
          className="absolute right-2.5 top-1/2 z-10 grid size-7 -translate-y-1/2 place-items-center rounded-full text-muted-foreground transition-colors duration-200 hover:bg-zinc-100 hover:text-foreground"
        >
          <Icon icon={Cancel01Icon} size={14} aria-hidden />
        </button>
      ) : null}
    </div>
  );
}
