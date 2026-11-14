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
    assistantName,
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
    <p className="text-body text-foreground-secondary">
      Paid with your Link wallet. {assistantName} never sees your card number.
    </p>
  );
  if (readOnly) return walletNote;
  if (isLoading) {
    return (
      <p className="text-body text-foreground-secondary">
        Loading the purchase…
      </p>
    );
  }
  if (loadFailed) {
    return (
      <div className="flex flex-col gap-2">
        <p role="alert" className="text-body text-error-foreground">
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
        assistantName={assistantName}
        agentNotTold={agentNotTold}
        onTellAgent={retryTellAgent}
      />
    );
  }
  if (!state || !purchase) {
    return (
      <p className="text-body text-foreground-secondary">
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
        <p role="alert" className="text-body text-error-foreground">
          {error}
        </p>
      )}
    </div>
  );
}
