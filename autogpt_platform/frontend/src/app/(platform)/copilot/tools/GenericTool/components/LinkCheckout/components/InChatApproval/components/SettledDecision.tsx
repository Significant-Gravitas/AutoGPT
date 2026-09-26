import { Button } from "@/components/atoms/Button/Button";

const SETTLED: Record<string, string> = {
  approved: "Approved. Completing the purchase…",
  declined: "Declined. Nothing was charged.",
  expired: "This purchase request expired. Nothing was charged.",
};

interface Props {
  state: string;
  agentNotTold: boolean;
  onTellAgent: () => void;
}

export function SettledDecision({ state, agentNotTold, onTellAgent }: Props) {
  return (
    <div className="flex flex-col gap-2">
      <p className="text-sm text-zinc-600">{SETTLED[state]}</p>
      {agentNotTold && (
        <>
          <p role="alert" className="text-sm text-red-600">
            Your decision is saved, but the message to the agent did not send.
          </p>
          <Button size="small" variant="secondary" onClick={onTellAgent}>
            Tell the agent
          </Button>
        </>
      )}
    </div>
  );
}
