"use client";

import type { InvitedUserResponse } from "@/app/api/__generated__/models/invitedUserResponse";
import { Badge } from "@/components/atoms/Badge/Badge";
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
  invitedUsers: InvitedUserResponse[];
  isLoading: boolean;
  isRefreshing: boolean;
  pendingInviteAction: string | null;
  onRetryTally: (invitedUserId: string) => void;
  onRevoke: (invitedUserId: string) => void;
}

function getInviteBadgeVariant(status: InvitedUserResponse["status"]) {
  if (status === "CLAIMED") {
    return "success";
  }

  if (status === "REVOKED") {
    return "error";
  }

  return "info";
}

function getTallyBadgeVariant(status: InvitedUserResponse["tally_status"]) {
  if (status === "READY") {
    return "success";
  }

  if (status === "FAILED") {
    return "error";
  }

  return "info";
}

function formatDate(value: Date | undefined) {
  if (!value) {
    return "-";
  }

  return value.toLocaleString();
}

function getTallySummary(invitedUser: InvitedUserResponse) {
  if (invitedUser.tally_status === "FAILED" && invitedUser.tally_error) {
    return invitedUser.tally_error;
  }

  if (invitedUser.tally_status === "READY" && invitedUser.tally_understanding) {
    return "Stored and ready for activation";
  }

  if (invitedUser.tally_status === "READY") {
    return "No matching Tally submission found";
  }

  if (invitedUser.tally_status === "RUNNING") {
    return "Extraction in progress";
  }

  return "Waiting to run";
}

function isActionPending(
  pendingInviteAction: string | null,
  action: "retry" | "revoke",
  invitedUserId: string,
) {
  return pendingInviteAction === `${action}:${invitedUserId}`;
}

export function InvitedUsersTable({
  invitedUsers,
  isLoading,
  isRefreshing,
  pendingInviteAction,
  onRetryTally,
  onRevoke,
}: Props) {
  return (
    <div className="flex flex-col gap-4">
      <div className="flex items-center justify-between gap-4">
        <div className="flex flex-col gap-1">
          <h2 className="text-xl font-semibold text-zinc-900">Invited users</h2>
          <p className="text-sm text-zinc-600">
            Live invite state, claim status, and Tally pre-seeding progress.
          </p>
        </div>
        <span className="text-xs uppercase tracking-[0.18em] text-zinc-400">
          {isRefreshing ? "Refreshing" : `${invitedUsers.length} total`}
        </span>
      </div>

      <div className="overflow-hidden rounded-2xl border border-zinc-200">
        <Table>
          <TableHeader className="bg-zinc-50">
            <TableRow>
              <TableHead>Email</TableHead>
              <TableHead>Name</TableHead>
              <TableHead>Invite</TableHead>
              <TableHead>Tally</TableHead>
              <TableHead>Claimed User</TableHead>
              <TableHead>Created</TableHead>
              <TableHead className="text-right">Actions</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {isLoading ? (
              <TableRow>
                <TableCell
                  colSpan={7}
                  className="py-10 text-center text-zinc-500"
                >
                  Loading invited users...
                </TableCell>
              </TableRow>
            ) : invitedUsers.length === 0 ? (
              <TableRow>
                <TableCell
                  colSpan={7}
                  className="py-10 text-center text-zinc-500"
                >
                  No invited users yet
                </TableCell>
              </TableRow>
            ) : (
              invitedUsers.map((invitedUser) => (
                <TableRow key={invitedUser.id} className="align-top">
                  <TableCell className="font-medium text-zinc-900">
                    {invitedUser.email}
                  </TableCell>
                  <TableCell>{invitedUser.name || "-"}</TableCell>
                  <TableCell>
                    <Badge variant={getInviteBadgeVariant(invitedUser.status)}>
                      {invitedUser.status}
                    </Badge>
                  </TableCell>
                  <TableCell>
                    <div className="flex max-w-xs flex-col gap-2">
                      <Badge
                        variant={getTallyBadgeVariant(invitedUser.tally_status)}
                      >
                        {invitedUser.tally_status}
                      </Badge>
                      <span className="text-xs text-zinc-500">
                        {getTallySummary(invitedUser)}
                      </span>
                      <span className="text-xs text-zinc-400">
                        {formatDate(invitedUser.tally_computed_at ?? undefined)}
                      </span>
                    </div>
                  </TableCell>
                  <TableCell className="font-mono text-xs text-zinc-500">
                    {invitedUser.auth_user_id || "-"}
                  </TableCell>
                  <TableCell className="text-sm text-zinc-500">
                    {formatDate(invitedUser.created_at)}
                  </TableCell>
                  <TableCell>
                    <div className="flex justify-end gap-2">
                      <Button
                        variant="outline"
                        size="small"
                        disabled={invitedUser.status === "REVOKED"}
                        loading={isActionPending(
                          pendingInviteAction,
                          "retry",
                          invitedUser.id,
                        )}
                        onClick={() => onRetryTally(invitedUser.id)}
                      >
                        Retry Tally
                      </Button>
                      <Button
                        variant="secondary"
                        size="small"
                        disabled={invitedUser.status !== "INVITED"}
                        loading={isActionPending(
                          pendingInviteAction,
                          "revoke",
                          invitedUser.id,
                        )}
                        onClick={() => onRevoke(invitedUser.id)}
                      >
                        Revoke
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
