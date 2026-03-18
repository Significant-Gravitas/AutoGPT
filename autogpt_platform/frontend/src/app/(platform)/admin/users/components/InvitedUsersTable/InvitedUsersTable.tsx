"use client";

import type { InvitedUserResponse } from "@/app/api/__generated__/models/invitedUserResponse";
import type { Pagination } from "@/app/api/__generated__/models/pagination";
import { Badge } from "@/components/atoms/Badge/Badge";
import { Button } from "@/components/atoms/Button/Button";
import { Input } from "@/components/atoms/Input/Input";
import { CaretLeft, CaretRight, MagnifyingGlass } from "@phosphor-icons/react";

interface Props {
  invitedUsers: InvitedUserResponse[];
  pagination: Pagination | null;
  currentPage: number;
  searchQuery: string;
  isLoading: boolean;
  isRefreshing: boolean;
  pendingInviteAction: string | null;
  onPageChange: (page: number) => void;
  onSearchChange: (value: string) => void;
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
  pagination,
  currentPage,
  searchQuery,
  isLoading,
  isRefreshing,
  pendingInviteAction,
  onPageChange,
  onSearchChange,
  onRetryTally,
  onRevoke,
}: Props) {
  const totalItems = pagination?.total_items ?? invitedUsers.length;
  const totalPages = pagination?.total_pages ?? 1;

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
          {isRefreshing ? "Refreshing" : `${totalItems} total`}
        </span>
      </div>

      <div className="relative">
        <MagnifyingGlass
          size={16}
          className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 text-zinc-400"
        />
        <Input
          id="invite-search"
          label="Search invited users"
          hideLabel
          type="text"
          placeholder="Search by email or name..."
          value={searchQuery}
          onChange={(e) => onSearchChange(e.target.value)}
          className="pl-9"
        />
      </div>

      <div className="overflow-hidden rounded-2xl border border-zinc-200">
        <table className="w-full text-sm">
          <thead className="bg-zinc-50">
            <tr className="border-b border-zinc-200">
              <th className="h-10 px-4 text-left font-medium text-zinc-600">
                Email
              </th>
              <th className="h-10 px-4 text-left font-medium text-zinc-600">
                Name
              </th>
              <th className="h-10 px-4 text-left font-medium text-zinc-600">
                Invite
              </th>
              <th className="h-10 px-4 text-left font-medium text-zinc-600">
                Tally
              </th>
              <th className="h-10 px-4 text-left font-medium text-zinc-600">
                Claimed User
              </th>
              <th className="h-10 px-4 text-left font-medium text-zinc-600">
                Created
              </th>
              <th className="h-10 px-4 text-right font-medium text-zinc-600">
                Actions
              </th>
            </tr>
          </thead>
          <tbody>
            {isLoading ? (
              <tr className="border-b border-zinc-100">
                <td
                  colSpan={7}
                  className="px-4 py-10 text-center text-zinc-500"
                >
                  Loading invited users...
                </td>
              </tr>
            ) : invitedUsers.length === 0 ? (
              <tr className="border-b border-zinc-100">
                <td
                  colSpan={7}
                  className="px-4 py-10 text-center text-zinc-500"
                >
                  No invited users yet
                </td>
              </tr>
            ) : (
              invitedUsers.map((invitedUser) => (
                <tr
                  key={invitedUser.id}
                  className="border-b border-zinc-100 align-top last:border-b-0"
                >
                  <td className="px-4 py-3 align-top font-medium text-zinc-900">
                    {invitedUser.email}
                  </td>
                  <td className="px-4 py-3 align-top">
                    {invitedUser.name || "-"}
                  </td>
                  <td className="px-4 py-3 align-top">
                    <Badge variant={getInviteBadgeVariant(invitedUser.status)}>
                      {invitedUser.status}
                    </Badge>
                  </td>
                  <td className="px-4 py-3 align-top">
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
                  </td>
                  <td className="px-4 py-3 align-top font-mono text-xs text-zinc-500">
                    {invitedUser.auth_user_id || "-"}
                  </td>
                  <td className="px-4 py-3 align-top text-sm text-zinc-500">
                    {formatDate(invitedUser.created_at)}
                  </td>
                  <td className="px-4 py-3 align-top">
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
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>

      {totalPages > 1 && (
        <div className="flex items-center justify-center gap-4">
          <Button
            type="button"
            variant="outline"
            size="small"
            className="min-w-0 rounded-lg border-zinc-200 text-zinc-700 hover:border-zinc-200 hover:bg-zinc-50"
            disabled={currentPage <= 1}
            onClick={() => onPageChange(currentPage - 1)}
            leftIcon={<CaretLeft size={14} />}
          >
            Previous
          </Button>
          <span className="text-sm text-zinc-500">
            Page {currentPage} of {totalPages}
          </span>
          <Button
            type="button"
            variant="outline"
            size="small"
            className="min-w-0 rounded-lg border-zinc-200 text-zinc-700 hover:border-zinc-200 hover:bg-zinc-50"
            disabled={currentPage >= totalPages}
            onClick={() => onPageChange(currentPage + 1)}
            rightIcon={<CaretRight size={14} />}
          >
            Next
          </Button>
        </div>
      )}
    </div>
  );
}
