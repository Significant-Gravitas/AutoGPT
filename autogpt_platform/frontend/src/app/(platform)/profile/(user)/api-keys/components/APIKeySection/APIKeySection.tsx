"use client";

import type { APIKeyInfo } from "@/app/api/__generated__/models/aPIKeyInfo";
import { Badge } from "@/components/atoms/Badge/Badge";
import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { DataTable } from "@/components/molecules/DataTable/DataTable";
import type { DataTableColumn } from "@/components/molecules/DataTable/helpers";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/molecules/DropdownMenu/DropdownMenu";
import { MoreVerticalIcon } from "@hugeicons/core-free-icons";
import { useAPISection } from "./useAPISection";

export function APIKeysSection() {
  const { apiKeys, isLoading, isDeleting, handleRevokeKey } = useAPISection();

  const columns: DataTableColumn<APIKeyInfo>[] = [
    { key: "name", header: "Name", cell: (key) => key.name },
    {
      key: "key",
      header: "API Key",
      cell: (key) => (
        <div
          data-testid="api-key-id"
          className="rounded-md border border-border p-1 px-2 text-xs"
        >
          {`${key.head}******************${key.tail}`}
        </div>
      ),
    },
    {
      key: "status",
      header: "Status",
      cell: (key) => (
        <Badge variant={key.status === "ACTIVE" ? "success" : "error"}>
          {key.status}
        </Badge>
      ),
    },
    {
      key: "created",
      header: "Created",
      cell: (key) => new Date(key.created_at).toLocaleDateString(),
    },
    {
      key: "lastUsed",
      header: "Last Used",
      cell: (key) =>
        key.last_used_at
          ? new Date(key.last_used_at).toLocaleDateString()
          : "Never",
    },
    {
      key: "actions",
      header: <span className="sr-only">Actions</span>,
      align: "right",
      cell: (key) => (
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button
              data-testid="api-key-actions"
              variant="ghost"
              size="icon-sm"
              withTooltip={false}
              aria-label="API key actions"
            >
              <Icon icon={MoreVerticalIcon} size={16} />
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end">
            <DropdownMenuItem
              className="text-destructive"
              onClick={() => handleRevokeKey(key.id)}
              disabled={isDeleting}
            >
              Revoke
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      ),
    },
  ];

  if (!isLoading && !apiKeys?.length) return null;

  return (
    <DataTable
      columns={columns}
      rows={apiKeys ?? []}
      getRowKey={(key) => key.id}
      caption="API keys"
      isLoading={isLoading}
      loadingRowCount={3}
    />
  );
}
