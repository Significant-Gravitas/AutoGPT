import { Button } from "@/components/atoms/Button/Button";

const SETTLED: Record<string, string> = {
  approved: "Approved. Completing the purchase…",
  declined: "Declined. Nothing was charged.",
  expired: "This purchase request expired. Nothing was charged.",
};

interface Props {
  state: string;
  assistantName: string;
  agentNotTold: boolean;
  onTellAgent: () => void;
}

export function SettledDecision({
  state,
  assistantName,
  agentNotTold,
  onTellAgent,
}: Props) {
  return (
    <div className="flex flex-col gap-2">
      <p className="text-body text-foreground-secondary">{SETTLED[state]}</p>
      {agentNotTold && (
        <>
          <p role="alert" className="text-body text-error-foreground">
            Your decision is saved, but the message to {assistantName} did not
            send.
          </p>
          <Button size="small" variant="secondary" onClick={onTellAgent}>
            {`Tell ${assistantName}`}
          </Button>
        </>
      )}
    </div>
  );
}
