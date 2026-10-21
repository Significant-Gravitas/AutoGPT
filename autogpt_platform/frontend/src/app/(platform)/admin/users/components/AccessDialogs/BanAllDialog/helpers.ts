import type { AccountIPReport } from "@/app/api/__generated__/models/accountIPReport";
import type { AuthAccessBlockKind } from "@/app/api/__generated__/models/authAccessBlockKind";
import type { RelatedAccount } from "@/app/api/__generated__/models/relatedAccount";
import { type AdminUser, pluralize } from "../../../helpers";
import type { ChecklistAccount } from "../AccountChecklist";

export interface BanAllAccount extends ChecklistAccount {
  role: string | null;
  // Linked by an address (or this account itself). One linked by email domain
  // alone may share only a mail provider the public list misses.
  isTickedByDefault: boolean;
}

interface Args {
  user: AdminUser;
  report: AccountIPReport;
  signedInUserId: string | undefined;
}

// This account first, then the related ones; never the admin's own account.
export function getBanAllAccounts({ user, report, signedInUserId }: Args) {
  const self: BanAllAccount = {
    id: user.id,
    email: user.email,
    name: user.name,
    banned: user.banned,
    role: user.role,
    badges: ["This account"],
    isTickedByDefault: true,
  };
  const related = report.related.map(toBanAllAccount);
  const all = [self, ...related];
  const accounts = all.filter((account) => account.id !== signedInUserId);
  return { accounts, excludesSelf: accounts.length < all.length };
}

export function getLinkBadges(account: RelatedAccount) {
  return [
    ...(account.via_ips.length > 0 ? ["IP"] : []),
    ...(account.via_domain ? ["Domain"] : []),
  ];
}

export interface AlsoBlockOption {
  key: string;
  kind: AuthAccessBlockKind;
  value: string;
  label: string;
  detail: string;
  isDisabled: boolean;
  disabledReason?: string;
}

// What a block-only confirm names: "2 addresses", "example.com", or both.
export function describeBlocks(options: AlsoBlockOption[]) {
  const addresses = options.filter((option) => option.kind === "IP").length;
  const domain = options.find((option) => option.kind === "EMAIL_DOMAIN");
  return [
    addresses > 0 ? pluralize(addresses, "address", "addresses") : null,
    domain?.label ?? null,
  ]
    .filter(Boolean)
    .join(" and ");
}

export function getAlsoBlockOptions(
  report: AccountIPReport,
): AlsoBlockOption[] {
  const ips = report.ips
    .filter((ip) => !ip.block)
    .map((ip) => ({
      key: `IP:${ip.ip_address}`,
      kind: "IP" as const,
      value: ip.ip_address,
      label: ip.display_address,
      detail: [
        "sign-up and sign-in refused",
        ip.other_account_count > 0
          ? `${pluralize(ip.other_account_count, "other account")} ${ip.other_account_count === 1 ? "uses" : "use"} it`
          : null,
      ]
        .filter(Boolean)
        .join(" · "),
      isDisabled: false,
    }));
  const { domain } = report;
  if (domain.block) return ips;
  return [
    ...ips,
    {
      key: `EMAIL_DOMAIN:${domain.value}`,
      kind: "EMAIL_DOMAIN" as const,
      value: domain.value,
      label: domain.value,
      detail: "sign-up refused",
      isDisabled: domain.is_public_provider,
      disabledReason: domain.is_public_provider
        ? "public mail provider"
        : undefined,
    },
  ];
}

function toBanAllAccount(account: RelatedAccount): BanAllAccount {
  return {
    id: account.id,
    email: account.email,
    name: account.name,
    banned: account.banned,
    role: account.role ?? null,
    badges: getLinkBadges(account),
    isTickedByDefault: account.via_ips.length > 0,
  };
}
