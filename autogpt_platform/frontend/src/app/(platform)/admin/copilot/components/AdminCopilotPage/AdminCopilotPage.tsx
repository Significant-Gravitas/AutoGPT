"use client";

import { Card } from "@/components/atoms/Card/Card";
import { Input } from "@/components/atoms/Input/Input";
import { CopilotUsersTable } from "../CopilotUsersTable/CopilotUsersTable";
import { useAdminCopilotPage } from "../../useAdminCopilotPage";
import { EmailSweepCard } from "./EmailSweepCard";
import { LatestSessionCard } from "./LatestSessionCard";
import { SelectedUserCard } from "./SelectedUserCard";
import { TriggerFlowsCard } from "./TriggerFlowsCard";

export function AdminCopilotPage() {
  const {
    search,
    selectedUser,
    pendingTriggerType,
    lastTriggeredSession,
    lastEmailSweepResult,
    searchedUsers,
    searchErrorMessage,
    isSearchingUsers,
    isRefreshingUsers,
    isTriggeringSession,
    isSendingPendingEmails,
    hasSearch,
    setSearch,
    handleSelectUser,
    handleSendPendingEmails,
    handleTriggerSession,
  } = useAdminCopilotPage();

  return (
    <div className="mx-auto flex max-w-7xl flex-col gap-6 p-6">
      <div className="flex flex-col gap-2">
        <h1 className="text-3xl font-bold text-zinc-900">Copilot</h1>
        <p className="max-w-3xl text-sm text-zinc-600">
          Manually create CTA, Nightly, or Callback Copilot sessions for a
          specific user. These controls bypass the normal eligibility checks so
          you can test each flow directly.
        </p>
      </div>

      <div className="grid gap-6 xl:grid-cols-[minmax(0,1.35fr),24rem]">
        <Card className="border border-zinc-200 shadow-sm">
          <div className="flex flex-col gap-4">
            <Input
              id="copilot-user-search"
              label="Search users"
              hint="Results update as you type"
              placeholder="Search by email, name, or user ID"
              value={search}
              onChange={(event) => setSearch(event.target.value)}
            />
            {searchErrorMessage ? (
              <p className="-mt-2 text-sm text-red-500">{searchErrorMessage}</p>
            ) : null}
            <CopilotUsersTable
              users={searchedUsers}
              isLoading={isSearchingUsers}
              isRefreshing={isRefreshingUsers}
              hasSearch={hasSearch}
              selectedUserId={selectedUser?.id ?? null}
              onSelectUser={handleSelectUser}
            />
          </div>
        </Card>

        <div className="flex flex-col gap-6">
          <SelectedUserCard selectedUser={selectedUser} />

          <TriggerFlowsCard
            disabled={!selectedUser}
            isTriggeringSession={isTriggeringSession}
            pendingTriggerType={pendingTriggerType}
            onTriggerSession={handleTriggerSession}
          />

          <EmailSweepCard
            userEmail={selectedUser?.email ?? null}
            disabled={!selectedUser}
            isSendingPendingEmails={isSendingPendingEmails}
            lastEmailSweepResult={lastEmailSweepResult}
            onSendPendingEmails={handleSendPendingEmails}
          />

          {selectedUser && lastTriggeredSession ? (
            <LatestSessionCard
              userEmail={selectedUser.email}
              session={lastTriggeredSession}
            />
          ) : null}
        </div>
      </div>
    </div>
  );
}
