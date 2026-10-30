"use client";

import { MoreVerticalIcon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";
import { LoadingSpinner } from "@/components/atoms/LoadingSpinner/LoadingSpinner";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/molecules/Table/components/TablePrimitives";
import { Badge } from "@/components/atoms/Badge/Badge";
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
          <LoadingSpinner variant="circle" size="medium" />
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
                    <div className="rounded-md border p-1 px-2">
                      <Text variant="small-tight" as="div" tone="inherit">
                        {`${key.head}******************${key.tail}`}
                      </Text>
                    </div>
                  </TableCell>
                  <TableCell>
                    <Badge
                      variant={
                        key.status === "ACTIVE" ? "solid" : "destructive"
                      }
                      className={
                        key.status === "ACTIVE"
                          ? // oxlint-disable-next-line design-system/no-restyle -- Badge has no bordered success chip yet; its success tint is a different shape and weight
                            "border-success bg-success/30 text-success-foreground"
                          : // oxlint-disable-next-line design-system/no-restyle -- Badge has no bordered error chip yet; its error tint is a different shape and weight
                            "border-error-hover bg-error-surface text-error-foreground"
                      }
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
                          variant="compact-ghost"
                          size="compact-sm"
                        >
                          <Icon icon={MoreVerticalIcon} className="h-4 w-4" />
                        </Button>
                      </DropdownMenuTrigger>
                      <DropdownMenuContent align="end">
                        <DropdownMenuItem
                          onClick={() => handleRevokeKey(key.id)}
                          disabled={isDeleting}
                        >
                          <span className="text-destructive">Revoke</span>
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
