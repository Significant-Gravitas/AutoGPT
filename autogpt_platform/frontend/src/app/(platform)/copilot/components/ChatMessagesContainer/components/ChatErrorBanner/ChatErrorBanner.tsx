import { Button } from "@/components/atoms/Button/Button";

import {
  describeChatError,
  type ChatErrorAction,
} from "../../../../chatErrorGuidance";

interface Props {
  error: Error;
}

export function ChatErrorBanner({ error }: Props) {
  const { title, action } = describeChatError(error);
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
      {action ? (
        <Button
          as="NextLink"
          href={hrefFor(action)}
          variant="secondary"
          size="small"
          className="mt-3"
        >
          {action === "sign-in" ? "Sign in" : "Start a new chat"}
        </Button>
      ) : null}
    </div>
  );
}

function hrefFor(action: ChatErrorAction) {
  if (action === "new-chat") return "/copilot";
  const here =
    typeof window === "undefined"
      ? "/copilot"
      : `${window.location.pathname}${window.location.search}`;
  return `/login?next=${encodeURIComponent(here)}`;
}
