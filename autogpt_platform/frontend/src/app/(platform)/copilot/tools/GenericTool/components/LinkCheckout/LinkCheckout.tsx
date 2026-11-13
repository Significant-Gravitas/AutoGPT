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
  const fromRecord = approvesInChat && !readOnly;

  return (
    <div className="flex flex-col gap-3 rounded-lg border border-border-base bg-surface p-4">
      <div className="flex items-start justify-between gap-4">
        <div>
          <p className="font-semibold text-foreground-strong">
            {fromRecord ? "Link purchase" : merchant}
          </p>
          {!fromRecord && (
            <p className="text-body text-foreground-secondary">{status}</p>
          )}
        </div>
        {!fromRecord && (
          <p className="text-lead font-semibold text-foreground-strong">
            {total}
          </p>
        )}
      </div>
      {testMode && (
        <p className="text-body font-medium text-foreground-secondary">
          Test payment · no charge
        </p>
      )}
      {approvesInChat && checkoutId && sessionId ? (
        <InChatApproval sessionId={sessionId} checkoutId={checkoutId} />
      ) : (
        <>
          {actionMessage && (
            <p className="text-body text-foreground-emphasis">
              {actionMessage}
            </p>
          )}
          <p className="text-body text-foreground-secondary">{message}</p>
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
            <p role="alert" className="text-body text-error-foreground">
              {error}
            </p>
          )}
        </>
      )}
    </div>
  );
}
