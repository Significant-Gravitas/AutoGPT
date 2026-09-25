import { Button } from "@/components/atoms/Button/Button";
import { PurchaseDetails } from "./components/PurchaseDetails";
import { useInChatApproval } from "./useInChatApproval";

interface Props {
  sessionId: string;
  checkoutId: string;
}

const SETTLED: Record<string, string> = {
  approved: "Approved. Completing the purchase…",
  declined: "Declined. Nothing was charged.",
  expired: "This purchase request expired. Nothing was charged.",
};

export function InChatApproval({ sessionId, checkoutId }: Props) {
  const {
    readOnly,
    state,
    purchase,
    isLoading,
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
  if (isLoading) return null;
  if (state && state !== "awaiting") {
    return <p className="text-sm text-zinc-600">{SETTLED[state]}</p>;
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
