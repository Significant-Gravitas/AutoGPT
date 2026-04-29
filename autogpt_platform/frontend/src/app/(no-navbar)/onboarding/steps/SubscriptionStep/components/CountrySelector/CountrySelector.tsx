"use client";

import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/molecules/Popover/Popover";
import { useMountEffect } from "@/hooks/useMountEffect";
import { cn } from "@/lib/utils";
import { MagnifyingGlass } from "@phosphor-icons/react";
import { useRef, useState } from "react";
import { COUNTRIES } from "../../countries";

interface Props {
  selected: number;
  onSelect: (idx: number) => void;
}

export function CountrySelector({ selected, onSelect }: Props) {
  const [open, setOpen] = useState(false);
  const country = COUNTRIES[selected];

  function handleOpenChange(next: boolean) {
    setOpen(next);
  }

  return (
    <Popover open={open} onOpenChange={handleOpenChange}>
      <PopoverTrigger asChild>
        <button
          type="button"
          className="flex min-w-[150px] items-center gap-1.5 px-2.5 py-1.5 text-sm text-zinc-600 transition-colors hover:text-zinc-900"
        >
          <span className="text-base">{country.flag}</span>
          <span>{country.name}</span>
          <span>({country.symbol.trim()})</span>
          <span className="ml-auto text-[10px] text-zinc-400">▾</span>
        </button>
      </PopoverTrigger>
      <PopoverContent
        side="top"
        align="end"
        sideOffset={8}
        collisionPadding={16}
        className="flex max-h-[340px] w-[260px] max-w-[calc(100vw-2rem)] flex-col overflow-hidden rounded-xl border-0 bg-zinc-900 p-0 text-white shadow-xl"
      >
        <CountryList
          selected={selected}
          onSelect={(idx) => {
            onSelect(idx);
            handleOpenChange(false);
          }}
        />
      </PopoverContent>
    </Popover>
  );
}

interface ListProps {
  selected: number;
  onSelect: (idx: number) => void;
}

function CountryList({ selected, onSelect }: ListProps) {
  const [search, setSearch] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);

  useMountEffect(() => {
    inputRef.current?.focus();
  });

  const query = search.toLowerCase();
  const filtered = COUNTRIES.map((c, idx) => ({ c, idx })).filter(
    ({ c }) =>
      c.name.toLowerCase().includes(query) ||
      c.countryCode.toLowerCase().includes(query) ||
      c.currencyCode.toLowerCase().includes(query),
  );

  return (
    <>
      <div className="px-3 pb-1.5 pt-2.5">
        <div className="flex items-center gap-2 rounded-lg border border-white/10 bg-white/5 px-2.5 py-1.5">
          <MagnifyingGlass
            size={12}
            className="shrink-0 text-white/40"
            weight="bold"
          />
          <input
            ref={inputRef}
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search country…"
            aria-label="Search country"
            className="w-full bg-transparent text-xs text-white outline-none placeholder:text-white/35"
          />
        </div>
      </div>

      <div className="max-h-[270px] overflow-y-auto px-0 py-1">
        {filtered.length === 0 && (
          <div className="px-3.5 py-3 text-xs text-white/40">
            No matching countries
          </div>
        )}
        {filtered.map(({ c, idx }) => (
          <button
            key={c.countryCode}
            type="button"
            onClick={() => onSelect(idx)}
            className={cn(
              "flex w-full items-center justify-between px-3.5 py-2 text-xs text-white transition-colors",
              idx === selected ? "bg-purple-500/15" : "hover:bg-white/[0.08]",
            )}
          >
            <span>
              {c.flag}&nbsp;&nbsp;{c.name}
            </span>
            <span className="text-[10px] text-white/35">{c.currencyCode}</span>
          </button>
        ))}
      </div>
    </>
  );
}
