"use client";

import { useState, useEffect } from "react";
import { useRouter, usePathname, useSearchParams } from "next/navigation";
import { CreditTransactionType } from "@/lib/autogpt-server-api";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/__legacy__/ui/select";
import { AdminUserSearch } from "../../components/AdminUserSearch";

export function SearchAndFilterAdminSpending({
  initialSearch,
}: {
  initialSearch?: string;
}) {
  const router = useRouter();
  const pathname = usePathname();
  const searchParams = useSearchParams();

  // Initialize state from URL parameters
  const [searchQuery, setSearchQuery] = useState(initialSearch || "");
  const [selectedStatus, setSelectedStatus] = useState<string>(
    searchParams.get("status") || "ALL",
  );

  // Update local state when URL parameters change
  useEffect(() => {
    const status = searchParams.get("status");
    setSelectedStatus(status || "ALL");
    setSearchQuery(searchParams.get("search") || "");
  }, [searchParams]);

  function handleSearch(query: string) {
    const params = new URLSearchParams(searchParams.toString());

    if (query) {
      params.set("search", query);
    } else {
      params.delete("search");
    }

    if (selectedStatus !== "ALL") {
      params.set("status", selectedStatus);
    } else {
      params.delete("status");
    }

    params.set("page", "1"); // Reset to first page on new search

    router.push(`${pathname}?${params.toString()}`);
  }

  return (
    <div className="flex items-center justify-between">
      <AdminUserSearch
        value={searchQuery}
        onChange={setSearchQuery}
        onSearch={handleSearch}
      />

      <Select
        value={selectedStatus}
        onValueChange={(value: string) => {
          setSelectedStatus(value);
          const params = new URLSearchParams(searchParams.toString());
          if (value === "ALL") {
            params.delete("status");
          } else {
            params.set("status", value);
          }
          params.set("page", "1");
          router.push(`${pathname}?${params.toString()}`);
        }}
      >
        <SelectTrigger className="w-1/4">
          <SelectValue placeholder="Select Status" />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value="ALL">All</SelectItem>
          <SelectItem value={CreditTransactionType.TOP_UP}>Top Up</SelectItem>
          <SelectItem value={CreditTransactionType.USAGE}>Usage</SelectItem>
          <SelectItem value={CreditTransactionType.REFUND}>Refund</SelectItem>
          <SelectItem value={CreditTransactionType.GRANT}>Grant</SelectItem>
          <SelectItem value={CreditTransactionType.CARD_CHECK}>
            Card Check
          </SelectItem>
        </SelectContent>
      </Select>
    </div>
  );
}
