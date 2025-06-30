"use client";
import { Input } from "@/components/ui/input";
import { Search, X } from "lucide-react";
import { useLibrarySearchbar } from "./useLibrarySearchbar";

export default function LibrarySearchBar(): React.ReactNode {
  const { handleSearchInput, handleClear, setIsFocused, isFocused, inputRef } =
    useLibrarySearchbar();
  return (
    <div
      onClick={() => inputRef.current?.focus()}
      className="relative z-[21] mx-auto flex h-[50px] w-full max-w-[500px] flex-1 cursor-pointer items-center rounded-[45px] bg-[#EDEDED] px-[24px] py-[10px]"
    >
      <Search
        className="mr-2 h-[29px] w-[29px] text-neutral-900"
        strokeWidth={1.25}
      />

      <Input
        ref={inputRef}
        onFocus={() => setIsFocused(true)}
        onBlur={() => !inputRef.current?.value && setIsFocused(false)}
        onChange={handleSearchInput}
        className="flex-1 border-none font-sans text-[16px] font-normal leading-7 shadow-none focus:shadow-none focus:ring-0"
        type="text"
        placeholder="Search agents"
      />

      {isFocused && inputRef.current?.value && (
        <X
          className="ml-2 h-[29px] w-[29px] cursor-pointer text-neutral-900"
          strokeWidth={1.25}
          onClick={handleClear}
        />
      )}
    </div>
  );
}
