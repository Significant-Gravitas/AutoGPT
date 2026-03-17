import type { TriggerCopilotSessionResponse } from "@/app/api/__generated__/models/triggerCopilotSessionResponse";
import { Badge } from "@/components/atoms/Badge/Badge";
import { Button } from "@/components/atoms/Button/Button";
import { Card } from "@/components/atoms/Card/Card";
import { getSessionStartTypeLabel } from "@/app/(platform)/copilot/helpers";

interface Props {
  userEmail: string;
  session: TriggerCopilotSessionResponse;
}

export function LatestSessionCard({ userEmail, session }: Props) {
  return (
    <Card className="border border-zinc-200 shadow-sm">
      <div className="flex flex-col gap-4">
        <div className="flex items-center justify-between gap-3">
          <h2 className="text-xl font-semibold text-zinc-900">
            Latest session
          </h2>
          <Badge variant="success">
            {getSessionStartTypeLabel(session.start_type) ?? session.start_type}
          </Badge>
        </div>
        <p className="text-sm text-zinc-600">
          A new Copilot session was created for {userEmail}.
        </p>
        <div>
          <span className="text-xs uppercase tracking-[0.18em] text-zinc-400">
            Session ID
          </span>
          <p className="mt-1 break-all font-mono text-xs text-zinc-500">
            {session.session_id}
          </p>
        </div>
        <Button
          as="NextLink"
          href={`/copilot?sessionId=${session.session_id}`}
          target="_blank"
          rel="noreferrer"
        >
          Open session
        </Button>
      </div>
    </Card>
  );
}
