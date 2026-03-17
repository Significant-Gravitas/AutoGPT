import type { AdminCopilotUserSummary } from "@/app/api/__generated__/models/adminCopilotUserSummary";
import { Badge } from "@/components/atoms/Badge/Badge";
import { Card } from "@/components/atoms/Card/Card";

interface Props {
  selectedUser: AdminCopilotUserSummary | null;
}

export function SelectedUserCard({ selectedUser }: Props) {
  return (
    <Card className="border border-zinc-200 shadow-sm">
      <div className="flex flex-col gap-4">
        <div className="flex items-center justify-between gap-3">
          <h2 className="text-xl font-semibold text-zinc-900">Selected user</h2>
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
              <p className="mt-1">{selectedUser.name || "No display name"}</p>
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
  );
}
