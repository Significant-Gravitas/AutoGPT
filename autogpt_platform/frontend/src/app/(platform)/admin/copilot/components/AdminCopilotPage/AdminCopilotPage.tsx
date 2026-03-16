"use client";

import { ChatSessionStartType } from "@/app/api/__generated__/models/chatSessionStartType";
import { Badge } from "@/components/atoms/Badge/Badge";
import { Button } from "@/components/atoms/Button/Button";
import { Card } from "@/components/atoms/Card/Card";
import { Input } from "@/components/atoms/Input/Input";
import { CopilotUsersTable } from "../CopilotUsersTable/CopilotUsersTable";
import { useAdminCopilotPage } from "../../useAdminCopilotPage";

function getStartTypeLabel(startType: ChatSessionStartType) {
  if (startType === ChatSessionStartType.AUTOPILOT_INVITE_CTA) {
    return "CTA";
  }

  if (startType === ChatSessionStartType.AUTOPILOT_NIGHTLY) {
    return "Nightly";
  }

  if (startType === ChatSessionStartType.AUTOPILOT_CALLBACK) {
    return "Callback";
  }

  return startType;
}

const triggerOptions = [
  {
    label: "Trigger CTA",
    description:
      "Runs the beta invite CTA flow even if the user would not normally qualify.",
    startType: ChatSessionStartType.AUTOPILOT_INVITE_CTA,
    variant: "primary" as const,
  },
  {
    label: "Trigger Nightly",
    description:
      "Runs the nightly proactive Autopilot flow immediately for the selected user.",
    startType: ChatSessionStartType.AUTOPILOT_NIGHTLY,
    variant: "outline" as const,
  },
  {
    label: "Trigger Callback",
    description:
      "Runs the callback re-engagement flow without checking the normal callback cohort.",
    startType: ChatSessionStartType.AUTOPILOT_CALLBACK,
    variant: "secondary" as const,
  },
];

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
          <Card className="border border-zinc-200 shadow-sm">
            <div className="flex flex-col gap-4">
              <div className="flex items-center justify-between gap-3">
                <h2 className="text-xl font-semibold text-zinc-900">
                  Selected user
                </h2>
                {selectedUser ? <Badge variant="info">Ready</Badge> : null}
              </div>

              {selectedUser ? (
                <div className="flex flex-col gap-3 text-sm text-zinc-600">
                  <div>
                    <span className="text-xs uppercase tracking-[0.18em] text-zinc-400">
                      Email
                    </span>
                    <p className="mt-1 font-medium text-zinc-900">
                      {selectedUser.email}
                    </p>
                  </div>
                  <div>
                    <span className="text-xs uppercase tracking-[0.18em] text-zinc-400">
                      Name
                    </span>
                    <p className="mt-1">
                      {selectedUser.name || "No display name"}
                    </p>
                  </div>
                  <div>
                    <span className="text-xs uppercase tracking-[0.18em] text-zinc-400">
                      Timezone
                    </span>
                    <p className="mt-1">{selectedUser.timezone}</p>
                  </div>
                  <div>
                    <span className="text-xs uppercase tracking-[0.18em] text-zinc-400">
                      User ID
                    </span>
                    <p className="mt-1 break-all font-mono text-xs text-zinc-500">
                      {selectedUser.id}
                    </p>
                  </div>
                </div>
              ) : (
                <p className="text-sm text-zinc-500">
                  Select a user from the results table to enable manual Copilot
                  triggers.
                </p>
              )}
            </div>
          </Card>

          <Card className="border border-zinc-200 shadow-sm">
            <div className="flex flex-col gap-4">
              <div className="flex flex-col gap-1">
                <h2 className="text-xl font-semibold text-zinc-900">
                  Trigger flows
                </h2>
                <p className="text-sm text-zinc-600">
                  Each action creates a new session immediately for the selected
                  user.
                </p>
              </div>

              <div className="flex flex-col gap-3">
                {triggerOptions.map((option) => (
                  <div
                    key={option.startType}
                    className="rounded-2xl border border-zinc-200 p-4"
                  >
                    <div className="flex flex-col gap-3">
                      <div className="flex flex-col gap-1">
                        <span className="font-medium text-zinc-900">
                          {option.label}
                        </span>
                        <p className="text-sm text-zinc-600">
                          {option.description}
                        </p>
                      </div>
                      <Button
                        variant={option.variant}
                        disabled={!selectedUser || isTriggeringSession}
                        loading={pendingTriggerType === option.startType}
                        onClick={() => handleTriggerSession(option.startType)}
                      >
                        {option.label}
                      </Button>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </Card>

          <Card className="border border-zinc-200 shadow-sm">
            <div className="flex flex-col gap-4">
              <div className="flex flex-col gap-1">
                <h2 className="text-xl font-semibold text-zinc-900">
                  Email follow-up
                </h2>
                <p className="text-sm text-zinc-600">
                  Run the pending Copilot completion-email sweep immediately for
                  the selected user.
                </p>
              </div>
              <Button
                variant="secondary"
                disabled={!selectedUser || isSendingPendingEmails}
                loading={isSendingPendingEmails}
                onClick={handleSendPendingEmails}
              >
                Send pending emails
              </Button>
              {selectedUser && lastEmailSweepResult ? (
                <div className="rounded-2xl border border-zinc-200 p-4 text-sm text-zinc-600">
                  <p className="font-medium text-zinc-900">
                    Last sweep for {selectedUser.email}
                  </p>
                  <div className="mt-3 grid gap-3 sm:grid-cols-2">
                    <div>
                      <span className="text-xs uppercase tracking-[0.18em] text-zinc-400">
                        Candidates
                      </span>
                      <p className="mt-1 text-zinc-900">
                        {lastEmailSweepResult.candidate_count}
                      </p>
                    </div>
                    <div>
                      <span className="text-xs uppercase tracking-[0.18em] text-zinc-400">
                        Processed
                      </span>
                      <p className="mt-1 text-zinc-900">
                        {lastEmailSweepResult.processed_count}
                      </p>
                    </div>
                    <div>
                      <span className="text-xs uppercase tracking-[0.18em] text-zinc-400">
                        Sent
                      </span>
                      <p className="mt-1 text-zinc-900">
                        {lastEmailSweepResult.sent_count}
                      </p>
                    </div>
                    <div>
                      <span className="text-xs uppercase tracking-[0.18em] text-zinc-400">
                        Skipped
                      </span>
                      <p className="mt-1 text-zinc-900">
                        {lastEmailSweepResult.skipped_count}
                      </p>
                    </div>
                    <div>
                      <span className="text-xs uppercase tracking-[0.18em] text-zinc-400">
                        Repairs queued
                      </span>
                      <p className="mt-1 text-zinc-900">
                        {lastEmailSweepResult.repair_queued_count}
                      </p>
                    </div>
                    <div>
                      <span className="text-xs uppercase tracking-[0.18em] text-zinc-400">
                        Running / failed
                      </span>
                      <p className="mt-1 text-zinc-900">
                        {lastEmailSweepResult.running_count} /{" "}
                        {lastEmailSweepResult.failed_count}
                      </p>
                    </div>
                  </div>
                </div>
              ) : null}
            </div>
          </Card>

          {selectedUser && lastTriggeredSession ? (
            <Card className="border border-zinc-200 shadow-sm">
              <div className="flex flex-col gap-4">
                <div className="flex items-center justify-between gap-3">
                  <h2 className="text-xl font-semibold text-zinc-900">
                    Latest session
                  </h2>
                  <Badge variant="success">
                    {getStartTypeLabel(lastTriggeredSession.start_type)}
                  </Badge>
                </div>
                <p className="text-sm text-zinc-600">
                  A new Copilot session was created for {selectedUser.email}.
                </p>
                <div>
                  <span className="text-xs uppercase tracking-[0.18em] text-zinc-400">
                    Session ID
                  </span>
                  <p className="mt-1 break-all font-mono text-xs text-zinc-500">
                    {lastTriggeredSession.session_id}
                  </p>
                </div>
                <Button
                  as="NextLink"
                  href={`/copilot?sessionId=${lastTriggeredSession.session_id}&showAutopilot=1`}
                  target="_blank"
                  rel="noreferrer"
                >
                  Open session
                </Button>
              </div>
            </Card>
          ) : null}
        </div>
      </div>
    </div>
  );
}
