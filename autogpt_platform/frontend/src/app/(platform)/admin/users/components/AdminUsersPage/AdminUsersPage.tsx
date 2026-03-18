"use client";

import { Card } from "@/components/atoms/Card/Card";
import { BulkInviteForm } from "../BulkInviteForm/BulkInviteForm";
import { InviteUserForm } from "../InviteUserForm/InviteUserForm";
import { InvitedUsersTable } from "../InvitedUsersTable/InvitedUsersTable";
import { useAdminUsersPage } from "../../useAdminUsersPage";

export function AdminUsersPage() {
  const {
    email,
    name,
    bulkInviteFile,
    bulkInviteInputKey,
    lastBulkInviteResult,
    invitedUsers,
    isLoadingInvitedUsers,
    isRefreshingInvitedUsers,
    isCreatingInvite,
    isBulkInviting,
    pendingInviteAction,
    setEmail,
    setName,
    handleBulkInviteFileChange,
    handleBulkInviteSubmit,
    handleCreateInvite,
    handleRetryTally,
    handleRevoke,
  } = useAdminUsersPage();

  return (
    <div className="mx-auto flex max-w-7xl flex-col gap-6 p-6">
      <div className="flex flex-col gap-2">
        <h1 className="text-3xl font-bold text-zinc-900">Beta Invites</h1>
        <p className="max-w-3xl text-sm text-zinc-600">
          Pre-provision beta users before they sign up. Invites store the
          platform-side record, run Tally understanding extraction, and activate
          the real account on the user&apos;s first authenticated request.
        </p>
      </div>

      <div className="grid gap-6 xl:grid-cols-[24rem,1fr]">
        <div className="flex flex-col gap-6">
          <Card className="border border-zinc-200 shadow-sm">
            <InviteUserForm
              email={email}
              name={name}
              isSubmitting={isCreatingInvite}
              onEmailChange={setEmail}
              onNameChange={setName}
              onSubmit={handleCreateInvite}
            />
          </Card>

          <Card className="border border-zinc-200 shadow-sm">
            <BulkInviteForm
              selectedFile={bulkInviteFile}
              inputKey={bulkInviteInputKey}
              isSubmitting={isBulkInviting}
              lastResult={lastBulkInviteResult}
              onFileChange={handleBulkInviteFileChange}
              onSubmit={handleBulkInviteSubmit}
            />
          </Card>
        </div>

        <Card className="border border-zinc-200 shadow-sm">
          <InvitedUsersTable
            invitedUsers={invitedUsers}
            isLoading={isLoadingInvitedUsers}
            isRefreshing={isRefreshingInvitedUsers}
            pendingInviteAction={pendingInviteAction}
            onRetryTally={handleRetryTally}
            onRevoke={handleRevoke}
          />
        </Card>
      </div>
    </div>
  );
}
