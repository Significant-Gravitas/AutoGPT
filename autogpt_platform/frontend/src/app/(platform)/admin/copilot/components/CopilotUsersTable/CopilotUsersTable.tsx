"use client";

import type { AdminCopilotUserSummary } from "@/app/api/__generated__/models/adminCopilotUserSummary";
import { Button } from "@/components/atoms/Button/Button";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/__legacy__/ui/table";

interface Props {
  users: AdminCopilotUserSummary[];
  isLoading: boolean;
  isRefreshing: boolean;
  hasSearch: boolean;
  selectedUserId: string | null;
  onSelectUser: (user: AdminCopilotUserSummary) => void;
}

function formatDate(value: Date) {
  return value.toLocaleString();
}

export function CopilotUsersTable({
  users,
  isLoading,
  isRefreshing,
  hasSearch,
  selectedUserId,
  onSelectUser,
}: Props) {
  let emptyMessage = "Search by email, name, or user ID to find a user.";
  if (hasSearch && isLoading) {
    emptyMessage = "Searching users...";
  } else if (hasSearch) {
    emptyMessage = "No matching users found.";
  }

  return (
    <div className="flex flex-col gap-4">
      <div className="flex items-center justify-between gap-4">
        <div className="flex flex-col gap-1">
          <h2 className="text-xl font-semibold text-zinc-900">User results</h2>
          <p className="text-sm text-zinc-600">
            Select an existing user, then run an Autopilot flow manually.
          </p>
        </div>
        <span className="text-xs uppercase tracking-[0.18em] text-zinc-400">
          {isRefreshing
            ? "Refreshing"
            : `${users.length} result${users.length === 1 ? "" : "s"}`}
        </span>
      </div>

      <div className="overflow-hidden rounded-2xl border border-zinc-200">
        <Table>
          <TableHeader className="bg-zinc-50">
            <TableRow>
              <TableHead>User</TableHead>
              <TableHead>Timezone</TableHead>
              <TableHead>Updated</TableHead>
              <TableHead className="text-right">Action</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {users.length === 0 ? (
              <TableRow>
                <TableCell
                  colSpan={4}
                  className="py-10 text-center text-zinc-500"
                >
                  {emptyMessage}
                </TableCell>
              </TableRow>
            ) : (
              users.map((user) => (
                <TableRow key={user.id} className="align-top">
                  <TableCell>
                    <div className="flex flex-col gap-1">
                      <span className="font-medium text-zinc-900">
                        {user.email}
                      </span>
                      <span className="text-sm text-zinc-600">
                        {user.name || "No display name"}
                      </span>
                      <span className="font-mono text-xs text-zinc-400">
                        {user.id}
                      </span>
                    </div>
                  </TableCell>
                  <TableCell className="text-sm text-zinc-600">
                    {user.timezone}
                  </TableCell>
                  <TableCell className="text-sm text-zinc-600">
                    {formatDate(user.updated_at)}
                  </TableCell>
                  <TableCell>
                    <div className="flex justify-end">
                      <Button
                        variant={
                          user.id === selectedUserId ? "secondary" : "outline"
                        }
                        size="small"
                        onClick={() => onSelectUser(user)}
                      >
                        {user.id === selectedUserId ? "Selected" : "Select"}
                      </Button>
                    </div>
                  </TableCell>
                </TableRow>
              ))
            )}
          </TableBody>
        </Table>
      </div>
    </div>
  );
}
