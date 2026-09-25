import { Button } from "@/components/atoms/Button/Button";
import { InChatApproval } from "./components/InChatApproval/InChatApproval";
import { checkoutPresentation } from "./helpers";
import { useLinkCheckout } from "./useLinkCheckout";

interface Props {
  output: Record<string, unknown>;
}

export function LinkCheckout({ output }: Props) {
  const {
    readOnly,
    checkoutId,
    sessionId,
    canContinue,
    sending,
    error,
    buttonLabel,
    handleContinue,
  } = useLinkCheckout(output);
  const {
    merchant,
    total,
    approvalUrl,
    linkLabel,
    status,
    message,
    actionMessage,
    testMode,
    approvesInChat,
  } = checkoutPresentation(output);

  return (
    <div className="flex flex-col gap-3 rounded-lg border border-zinc-200 bg-white p-4">
      <div className="flex items-start justify-between gap-4">
        <div>
          <p className="font-semibold text-zinc-900">{merchant}</p>
          <p className="text-sm text-zinc-600">{status}</p>
        </div>
        <p className="text-lg font-semibold text-zinc-900">{total}</p>
      </div>
      {testMode && (
        <p className="text-sm font-medium text-zinc-600">
          Test payment · no charge
        </p>
      )}
      {approvesInChat && checkoutId && sessionId ? (
        <InChatApproval sessionId={sessionId} checkoutId={checkoutId} />
      ) : (
        <>
          {actionMessage && (
            <p className="text-sm text-zinc-800">{actionMessage}</p>
          )}
          <p className="text-sm text-zinc-600">{message}</p>
          {approvalUrl && !readOnly && (
            <Button
              as="NextLink"
              href={approvalUrl}
              target="_blank"
              rel="noopener noreferrer"
            >
              {linkLabel}
            </Button>
          )}
          {canContinue && (
            <Button
              variant="outline"
              onClick={handleContinue}
              disabled={sending}
            >
              {sending ? "Checking Link…" : buttonLabel}
            </Button>
          )}
          {error && (
            <p role="alert" className="text-sm text-red-600">
              {error}
            </p>
          )}
        </>
      )}
    </div>
  );
}
