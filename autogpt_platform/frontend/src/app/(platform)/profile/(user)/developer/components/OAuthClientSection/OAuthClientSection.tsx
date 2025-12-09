"use client";

import { Loader2, MoreVertical } from "lucide-react";
import { Button } from "@/components/__legacy__/ui/button";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/__legacy__/ui/table";
import { Badge } from "@/components/__legacy__/ui/badge";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/__legacy__/ui/dropdown-menu";
import { useOAuthClientSection } from "./useOAuthClientSection";

export function OAuthClientSection() {
  const {
    oauthClients,
    isLoading,
    isDeleting,
    isSuspending,
    isActivating,
    handleDeleteClient,
    handleSuspendClient,
    handleActivateClient,
  } = useOAuthClientSection();

  const isActionPending = isDeleting || isSuspending || isActivating;

  return (
    <>
      {isLoading ? (
        <div className="flex justify-center p-4">
          <Loader2 className="h-6 w-6 animate-spin" />
        </div>
      ) : oauthClients && oauthClients.length > 0 ? (
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Name</TableHead>
              <TableHead>Client ID</TableHead>
              <TableHead>Type</TableHead>
              <TableHead>Status</TableHead>
              <TableHead>Created</TableHead>
              <TableHead></TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {oauthClients.map((client) => (
              <TableRow key={client.id} data-testid="oauth-client-row">
                <TableCell>
                  <div className="flex flex-col">
                    <span className="font-medium">{client.name}</span>
                    {client.description && (
                      <span className="text-xs text-muted-foreground">
                        {client.description}
                      </span>
                    )}
                  </div>
                </TableCell>
                <TableCell data-testid="oauth-client-id">
                  <div className="rounded-md border p-1 px-2 font-mono text-xs">
                    {client.client_id}
                  </div>
                </TableCell>
                <TableCell>
                  <Badge variant="outline">
                    {client.client_type === "confidential"
                      ? "Confidential"
                      : "Public"}
                  </Badge>
                </TableCell>
                <TableCell>
                  <Badge
                    variant={
                      client.status === "ACTIVE" ? "default" : "destructive"
                    }
                    className={
                      client.status === "ACTIVE"
                        ? "border-green-600 bg-green-100 text-green-800"
                        : "border-red-600 bg-red-100 text-red-800"
                    }
                  >
                    {client.status}
                  </Badge>
                </TableCell>
                <TableCell>
                  {new Date(client.created_at).toLocaleDateString()}
                </TableCell>
                <TableCell>
                  <DropdownMenu>
                    <DropdownMenuTrigger asChild>
                      <Button
                        data-testid="oauth-client-actions"
                        variant="ghost"
                        size="sm"
                      >
                        <MoreVertical className="h-4 w-4" />
                      </Button>
                    </DropdownMenuTrigger>
                    <DropdownMenuContent align="end">
                      {client.status === "ACTIVE" ? (
                        <DropdownMenuItem
                          onClick={() => handleSuspendClient(client.client_id)}
                          disabled={isActionPending}
                        >
                          Suspend
                        </DropdownMenuItem>
                      ) : (
                        <DropdownMenuItem
                          onClick={() => handleActivateClient(client.client_id)}
                          disabled={isActionPending}
                        >
                          Activate
                        </DropdownMenuItem>
                      )}
                      <DropdownMenuSeparator />
                      <DropdownMenuItem
                        className="text-destructive"
                        onClick={() => handleDeleteClient(client.client_id)}
                        disabled={isActionPending}
                      >
                        Delete
                      </DropdownMenuItem>
                    </DropdownMenuContent>
                  </DropdownMenu>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      ) : (
        <div className="py-8 text-center text-muted-foreground">
          No OAuth clients registered yet. Create one to get started.
        </div>
      )}
    </>
  );
}
