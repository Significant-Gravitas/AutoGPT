import type { ReactNode } from "react";

interface Props {
  children: ReactNode;
  locked: boolean;
  isSending: boolean;
  sent: boolean;
  error: string | null;
}

export function ResponseFrame(props: Props) {
  const { children, error } = props;
  const status = statusText(props);
  return (
    <section
      aria-label="Interactive response"
      className="my-3 min-w-0 space-y-3"
    >
      {children}
      {error && (
        <p role="alert" className="text-sm text-red-700">
          {error}
        </p>
      )}
      <p className="text-xs text-zinc-500" role="status">
        {status}
      </p>
    </section>
  );
}

function statusText({ locked, isSending, sent }: Props) {
  if (locked) return "Shared view · conversation actions are disabled";
  if (isSending) return "Sending to the conversation…";
  if (sent) return "Sent to the conversation";
  return "";
}
