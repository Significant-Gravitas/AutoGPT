"use client";

import { Loader2, MoreVertical } from "lucide-react";
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
          <Loader2 className="h-6 w-6 animate-spin" />
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
                    <Text
                      variant="small-tight"
                      as="div"
                      tone="inherit"
                      className="rounded-md border p-1 px-2"
                    >
                      {`${key.head}******************${key.tail}`}
                    </Text>
                  </TableCell>
                  <TableCell>
                    <Badge
                      variant={
                        key.status === "ACTIVE" ? "solid" : "destructive"
                      }
                      className={
                        key.status === "ACTIVE"
                          ? // TODO(tokens): no exact role (bg-green-100 #B7E2C7, text-green-800 #0C5A29)
                            "border-success bg-green-100 text-green-800"
                          : // TODO(tokens): no exact role (bg-red-100 #FDECEC, text-red-800 #832525)
                            "border-error-hover bg-red-100 text-red-800"
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
                          variant="legacy-ghost"
                          size="legacy-sm"
                        >
                          <MoreVertical className="h-4 w-4" />
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
