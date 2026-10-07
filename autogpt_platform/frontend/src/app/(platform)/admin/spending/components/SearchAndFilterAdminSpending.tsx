"use client";

import { useState, useEffect } from "react";
import { useRouter, usePathname, useSearchParams } from "next/navigation";
import { CreditTransactionType } from "@/lib/autogpt-server-api";
import { Select } from "@/components/atoms/Select/Select";
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
        id="spending-status-filter"
        label="Transaction status"
        hideLabel
        size="small"
        wrapperClassName="mb-0 w-1/4"
        placeholder="Select Status"
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
        options={[
          { value: "ALL", label: "All" },
          { value: CreditTransactionType.TOP_UP, label: "Top Up" },
          { value: CreditTransactionType.USAGE, label: "Usage" },
          { value: CreditTransactionType.REFUND, label: "Refund" },
          { value: CreditTransactionType.GRANT, label: "Grant" },
          { value: CreditTransactionType.CARD_CHECK, label: "Card Check" },
        ]}
      />
    </div>
  );
}
