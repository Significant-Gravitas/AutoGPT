"use client";

import { useState, useEffect } from "react";
import { useRouter, usePathname, useSearchParams } from "next/navigation";
import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Input } from "@/components/atoms/Input/Input";
import { Select } from "@/components/atoms/Select/Select";
import { Search01Icon } from "@hugeicons/core-free-icons";
import { SubmissionStatus } from "@/app/api/__generated__/models/submissionStatus";
import { isKey } from "@/lib/keyboard";

export function SearchAndFilterAdminMarketplace({
  initialSearch,
}: {
  initialStatus?: SubmissionStatus;
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

  const handleSearch = () => {
    const params = new URLSearchParams(searchParams.toString());

    if (searchQuery) {
      params.set("search", searchQuery);
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
  };

  return (
    <div className="flex items-center justify-between">
      <div className="flex w-full items-center gap-2">
        <Input
          id="admin-marketplace-search"
          label="Search agents by Name, Creator, or Description..."
          hideLabel
          size="small"
          wrapperClassName="mb-0"
          placeholder="Search agents by Name, Creator, or Description..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          onKeyDown={(e) => isKey(e, "Enter") && handleSearch()}
        />
        <Button
          variant="outline"
          size="small"
          onClick={handleSearch}
          aria-label="Search"
        >
          <Icon icon={Search01Icon} size={16} />
        </Button>
      </div>

      <Select
        id="admin-marketplace-status-filter"
        label="Status"
        hideLabel
        size="small"
        wrapperClassName="mb-0 w-[180px]"
        placeholder="Select Status"
        value={selectedStatus}
        onValueChange={(value) => {
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
          { value: SubmissionStatus.PENDING, label: "Pending" },
          { value: SubmissionStatus.APPROVED, label: "Approved" },
          { value: SubmissionStatus.REJECTED, label: "Rejected" },
        ]}
      />
    </div>
  );
}
