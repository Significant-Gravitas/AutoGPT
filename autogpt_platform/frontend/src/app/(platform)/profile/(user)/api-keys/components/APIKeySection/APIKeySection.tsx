"use client";

import { MoreVerticalIcon } from "@hugeicons/core-free-icons";
import { Button } from "@/components/atoms/Button/Button";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/__legacy__/ui/table";
import { Badge } from "@/components/atoms/Badge/Badge";
import { Icon } from "@/components/atoms/Icon/Icon";
import { LoadingSpinner } from "@/components/atoms/LoadingSpinner/LoadingSpinner";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/molecules/DropdownMenu/DropdownMenu";
import { useAPISection } from "./useAPISection";

export function APIKeysSection() {
  const { apiKeys, isLoading, isDeleting, handleRevokeKey } = useAPISection();

  return (
    <>
      {isLoading ? (
        <div className="flex justify-center p-4">
          <LoadingSpinner size="medium" />
        </div>
      ) : (
        apiKeys &&
        apiKeys.length > 0 && (
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Name</TableHead>
                <TableHead>API Key</TableHead>
                <TableHead>Status</TableHead>
                <TableHead>Created</TableHead>
                <TableHead>Last Used</TableHead>
                <TableHead></TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {apiKeys.map((key) => (
                <TableRow key={key.id} data-testid="api-key-row">
                  <TableCell>{key.name}</TableCell>
                  <TableCell data-testid="api-key-id">
                    <div className="rounded-md border p-1 px-2 text-xs">
                      {`${key.head}******************${key.tail}`}
                    </div>
                  </TableCell>
                  <TableCell>
                    <Badge
                      variant={key.status === "ACTIVE" ? "success" : "error"}
                    >
                      {key.status}
                    </Badge>
                  </TableCell>
                  <TableCell>
                    {new Date(key.created_at).toLocaleDateString()}
                  </TableCell>
                  <TableCell>
                    {key.last_used_at
                      ? new Date(key.last_used_at).toLocaleDateString()
                      : "Never"}
                  </TableCell>
                  <TableCell>
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
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        )
      )}
    </>
  );
}
