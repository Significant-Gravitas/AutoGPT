import { Button } from "@/components/atoms/Button/Button";
import { useIsBillingEnabled } from "@/hooks/useIsBillingEnabled";

import {
  describeChatError,
  type ChatErrorAction,
} from "../../../../chatErrorGuidance";

interface Props {
  error: Error;
}

export function ChatErrorBanner({ error }: Props) {
  const guidance = describeChatError(error);
  const isBillingEnabled = useIsBillingEnabled();
  // Without billing (most self-hosted installs) there are no plans to pick
  // and /settings/billing is a 404, so point at whoever runs the server.
  const noPlans = guidance.action === "plans" && !isBillingEnabled;
  const { title } = guidance;
  const action = noPlans ? null : guidance.action;
  const note = noPlans
    ? "Ask whoever runs this server to set one up."
    : guidance.note;
  return (
    <div
      data-testid="chat-error-banner"
      className="rounded-lg bg-red-50 p-4 text-sm text-red-700"
    >
      <details>
        <summary className="cursor-pointer font-medium">{title}</summary>
        <pre className="mt-2 max-h-40 overflow-auto whitespace-pre-wrap break-words text-xs text-red-600">
          {error.message}
        </pre>
      </details>
      {note ? <p className="mt-1 text-xs text-zinc-500">{note}</p> : null}
      {action ? (
        <Button
          as="NextLink"
          href={hrefFor(action)}
          variant="secondary"
          size="small"
          className="mt-3"
        >
          {ACTION_LABELS[action]}
        </Button>
      ) : null}
    </div>
  );
}

const ACTION_LABELS: Record<ChatErrorAction, string> = {
  "sign-in": "Sign in",
  "new-chat": "Start a new chat",
  plans: "See plans",
};

function hrefFor(action: ChatErrorAction) {
  if (action === "new-chat") return "/copilot";
  if (action === "plans") return "/settings/billing";
  const here =
    typeof window === "undefined"
      ? "/copilot"
      : `${window.location.pathname}${window.location.search}`;
  return `/login?next=${encodeURIComponent(here)}`;
}
