"use client";

import { useState } from "react";
import { Input } from "@/components/__legacy__/ui/input";
import { Button } from "@/components/atoms/Button/Button";
import { MagnifyingGlass } from "@phosphor-icons/react";

export interface AdminUserSearchProps {
  /** Current search query value (controlled). Falls back to internal state if omitted. */
  value?: string;
  /** Called when the input text changes */
  onChange?: (value: string) => void;
  /** Called when the user presses Enter or clicks the search button */
  onSearch: (query: string) => void;
  /** Placeholder text for the input */
  placeholder?: string;
  /** Disables the input and button while a search is in progress */
  isLoading?: boolean;
}

/**
 * Shared admin user search input.
 * Supports searching users by name, email, or partial/fuzzy text.
 * Can be used as controlled (value + onChange) or uncontrolled (internal state).
 */
export function AdminUserSearch({
  value: controlledValue,
  onChange,
  onSearch,
  placeholder = "Search users by Name or Email...",
  isLoading = false,
}: AdminUserSearchProps) {
  const [internalValue, setInternalValue] = useState("");

  const isControlled = controlledValue !== undefined;
  const currentValue = isControlled ? controlledValue : internalValue;

  function handleChange(newValue: string) {
    if (isControlled) {
      onChange?.(newValue);
    } else {
      setInternalValue(newValue);
    }
  }

  function handleSearch() {
    onSearch(currentValue.trim());
  }

  return (
    <div className="flex w-full items-center gap-2">
      <Input
        placeholder={placeholder}
        aria-label={placeholder}
        value={currentValue}
        onChange={(e) => handleChange(e.target.value)}
        onKeyDown={(e) => e.key === "Enter" && handleSearch()}
        disabled={isLoading}
      />
      <Button
        variant="outline"
        size="small"
        onClick={handleSearch}
        disabled={isLoading || !currentValue.trim()}
        loading={isLoading}
      >
        {isLoading ? "Searching..." : <MagnifyingGlass size={16} />}
      </Button>
    </div>
  );
}
