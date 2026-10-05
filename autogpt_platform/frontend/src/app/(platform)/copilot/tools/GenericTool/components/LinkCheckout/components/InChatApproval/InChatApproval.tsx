import { Button } from "@/components/atoms/Button/Button";
import { PurchaseDetails } from "./components/PurchaseDetails";
import { SettledDecision } from "./components/SettledDecision";
import { useInChatApproval } from "./useInChatApproval";

interface Props {
  sessionId: string;
  checkoutId: string;
}

export function InChatApproval({ sessionId, checkoutId }: Props) {
  const {
    readOnly,
    state,
    purchase,
    isLoading,
    loadFailed,
    retryLoad,
    agentNotTold,
    retryTellAgent,
    approving,
    declining,
    error,
    handleApprove,
    handleDecline,
  } = useInChatApproval({ sessionId, checkoutId });

  const walletNote = (
    <p className="text-sm text-zinc-600">
      Paid with your Link wallet. The agent never sees your card number.
    </p>
  );
  if (readOnly) return walletNote;
  if (isLoading) {
    return <p className="text-sm text-zinc-600">Loading the purchase…</p>;
  }
  if (loadFailed) {
    return (
      <div className="flex flex-col gap-2">
        <p role="alert" className="text-sm text-red-600">
          This purchase could not be loaded.
        </p>
        <Button size="small" variant="secondary" onClick={retryLoad}>
          Try again
        </Button>
      </div>
    );
  }
  if (state && state !== "awaiting") {
    return (
      <SettledDecision
        state={state}
        agentNotTold={agentNotTold}
        onTellAgent={retryTellAgent}
      />
    );
  }
  if (!state || !purchase) {
    return (
      <p className="text-sm text-zinc-600">
        This purchase is no longer waiting for approval.
      </p>
    );
  }

  return (
    <div className="flex flex-col gap-2">
      <PurchaseDetails {...purchase} />
      {walletNote}
      <div className="flex flex-wrap gap-2">
        <Button
          size="small"
          onClick={handleApprove}
          loading={approving}
          disabled={approving || declining}
        >
          {`Approve ${purchase.total}`.trim()}
        </Button>
        <Button
          size="small"
          variant="secondary"
          onClick={handleDecline}
          loading={declining}
          disabled={approving || declining}
        >
          Decline
        </Button>
      </div>
      {error && (
        <p role="alert" className="text-sm text-red-600">
          {error}
        </p>
      )}
    </div>
  );
}
