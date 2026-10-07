import { Text } from "@/components/atoms/Text/Text";
import type { CredentialRejection } from "@/app/api/__generated__/models/credentialRejection";

interface Props {
  rejection: CredentialRejection;
}

export function CredentialRejectionNotice({ rejection }: Props) {
  return (
    <div
      role="alert"
      className="rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-xs text-red-700"
    >
      {rejection.credential_title
        ? `"${rejection.credential_title}" was refused`
        : "The saved credential was refused"}
      {rejection.status_code ? ` — HTTP ${rejection.status_code}` : ""}
      <Text
        variant="small"
        as="p"
        unmask={false}
        className="mt-1 wrap-break-word text-inherit opacity-80"
      >
        {rejection.detail}
      </Text>
    </div>
  );
}
