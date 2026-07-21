"use client";

import { MagnifyingGlassIcon } from "@phosphor-icons/react";

interface Props {
  value: string;
  onChange: (next: string) => void;
  disabled?: boolean;
}

export function IntegrationsSearch({ value, onChange, disabled }: Props) {
  return (
    <div className="relative w-full">
      <MagnifyingGlassIcon
        size={20}
        className="pointer-events-none absolute left-4 top-1/2 -translate-y-1/2 text-[#83838C]"
      />
      <input
        type="text"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder="Search integrations..."
        aria-label="Search integrations"
        disabled={disabled}
        className="h-[46px] w-full rounded-3xl border border-[#DADADC] bg-white pl-12 pr-4 text-sm leading-[22px] text-[#1F1F20] placeholder:text-[#83838C] focus:border-purple-400 focus:outline-none focus:ring-1 focus:ring-purple-400 disabled:cursor-not-allowed disabled:opacity-60"
      />
    </div>
  );
}
