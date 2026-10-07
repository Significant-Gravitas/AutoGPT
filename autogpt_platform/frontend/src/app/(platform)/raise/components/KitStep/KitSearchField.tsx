"use client";

import { Button } from "@/components/atoms/Button/Button";
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
    <div className="relative w-full max-w-2xl">
      <Icon
        icon={Search01Icon}
        size={16}
        aria-hidden
        className="pointer-events-none absolute top-1/2 left-3.5 z-10 -translate-y-1/2 text-muted-foreground"
      />
      <Input
        id={`raise-${scope}-search`}
        label={label}
        hideLabel
        size="small"
        value={value}
        onChange={(event) => onChange(event.target.value)}
        placeholder={placeholder}
        className="pr-11 pl-10"
        wrapperClassName="mb-0 w-full [&_input]:h-10.5 [&_input]:py-3"
      />
      {/* The spinner replaces the clear button rather than sitting beside it:
          while a query is in flight there is nothing settled to clear yet. */}
      {isSearching ? (
        <Icon
          icon={Loading03Icon}
          size={16}
          aria-hidden
          className="absolute top-1/2 right-3.5 z-10 -translate-y-1/2 animate-spin text-muted-foreground motion-reduce:animate-none"
        />
      ) : value ? (
        <Button
          type="button"
          variant="ghost"
          size="icon-xs"
          withTooltip={false}
          onClick={() => onChange("")}
          aria-label="Clear search"
          className="absolute top-1/2 right-2.5 z-10 -translate-y-1/2 rounded-full border-0 text-muted-foreground duration-200 hover:bg-zinc-100 hover:text-foreground"
        >
          <Icon icon={Cancel01Icon} size={14} aria-hidden />
        </Button>
      ) : null}
    </div>
  );
}
