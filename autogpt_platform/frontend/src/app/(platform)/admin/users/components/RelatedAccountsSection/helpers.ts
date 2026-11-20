import type { AccountIPReport } from "@/app/api/__generated__/models/accountIPReport";

export function getNoRelatedMessage({ ips, domain }: AccountIPReport) {
  if (domain.is_public_provider) {
    return ips.length === 0
      ? "No addresses on record, so no accounts are linked."
      : "No other accounts share this account's addresses.";
  }
  return ips.length === 0
    ? "No addresses on record, so related accounts are matched by email domain only: none."
    : "No other accounts share this account's addresses or email domain.";
}
