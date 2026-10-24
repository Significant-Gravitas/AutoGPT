import type { CredentialRejection } from "@/app/api/__generated__/models/credentialRejection";

interface Props {
  rejection: CredentialRejection;
}

export function CredentialRejectionNotice({ rejection }: Props) {
  return (
    <div
      role="alert"
      // TODO(tokens): no exact role for border-red-200
      className="rounded-lg border border-red-200 bg-error-surface px-3 py-2 text-xs text-error-foreground"
    >
      {rejection.credential_title
        ? `"${rejection.credential_title}" was refused`
        : "The saved credential was refused"}
      {rejection.status_code ? ` — HTTP ${rejection.status_code}` : ""}
      <p className="mt-1 break-words opacity-80">{rejection.detail}</p>
    </div>
  );
}
