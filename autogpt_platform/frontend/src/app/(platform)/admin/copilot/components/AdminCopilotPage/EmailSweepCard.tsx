import type { SendCopilotEmailsResponse } from "@/app/api/__generated__/models/sendCopilotEmailsResponse";
import { Button } from "@/components/atoms/Button/Button";
import { Card } from "@/components/atoms/Card/Card";

interface Props {
  userEmail: string | null;
  disabled: boolean;
  isSendingPendingEmails: boolean;
  lastEmailSweepResult: SendCopilotEmailsResponse | null;
  onSendPendingEmails: () => void;
}

export function EmailSweepCard({
  userEmail,
  disabled,
  isSendingPendingEmails,
  lastEmailSweepResult,
  onSendPendingEmails,
}: Props) {
  return (
    <Card className="border border-zinc-200 shadow-sm">
      <div className="flex flex-col gap-4">
        <div className="flex flex-col gap-1">
          <h2 className="text-xl font-semibold text-zinc-900">
            Email follow-up
          </h2>
          <p className="text-sm text-zinc-600">
            Run the pending Copilot completion-email sweep immediately for the
            selected user.
          </p>
        </div>
        <Button
          variant="secondary"
          disabled={disabled || isSendingPendingEmails}
          loading={isSendingPendingEmails}
          onClick={onSendPendingEmails}
        >
          Send pending emails
        </Button>
        {userEmail && lastEmailSweepResult ? (
          <div className="rounded-2xl border border-zinc-200 p-4 text-sm text-zinc-600">
            <p className="font-medium text-zinc-900">
              Last sweep for {userEmail}
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
  );
}
