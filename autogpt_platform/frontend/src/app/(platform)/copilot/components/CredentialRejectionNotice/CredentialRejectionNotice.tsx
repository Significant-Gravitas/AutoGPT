import type { CredentialRejection } from "@/app/api/__generated__/models/credentialRejection";
import { AttentionCard } from "@/components/organisms/AttentionCard/AttentionCard";

interface Props {
  rejection: CredentialRejection;
}

export function CredentialRejectionNotice({ rejection }: Props) {
  return (
    <AttentionCard.Root variant="error" density="snug" role="alert">
      {rejection.credential_title
        ? `"${rejection.credential_title}" was refused`
        : "The saved credential was refused"}
      {rejection.status_code ? ` — HTTP ${rejection.status_code}` : ""}
      <p className="mt-1 break-words opacity-80">{rejection.detail}</p>
    </AttentionCard.Root>
  );
}
