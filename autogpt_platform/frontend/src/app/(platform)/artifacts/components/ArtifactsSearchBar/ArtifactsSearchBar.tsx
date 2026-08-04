"use client";

import { Input } from "@/components/atoms/Input/Input";
import { MagnifyingGlassIcon } from "@phosphor-icons/react";
import type { ChangeEvent } from "react";

interface Props {
  searchTerm: string;
  setSearchTerm: (value: string) => void;
}

export function ArtifactsSearchBar({ searchTerm, setSearchTerm }: Props) {
  function handleChange(event: ChangeEvent<HTMLInputElement>) {
    setSearchTerm(event.target.value);
  }

  return (
    <div
      data-testid="artifacts-search-bar"
      className="relative -mb-6 flex w-full items-center md:w-auto"
    >
      <MagnifyingGlassIcon
        width={18}
        height={18}
        className="absolute left-4 top-[34%] z-20 -translate-y-1/2 text-zinc-800"
      />
      <Input
        label="Search files"
        id="artifacts-search-bar"
        hideLabel
        type="text"
        value={searchTerm}
        onChange={handleChange}
        placeholder="Search files"
        className="min-w-[18rem] pl-12 lg:min-w-[24rem]"
        data-testid="artifacts-search-input"
      />
    </div>
  );
}
