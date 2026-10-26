"use client";

import type { RelatedAccount } from "@/app/api/__generated__/models/relatedAccount";
import { Badge } from "@/components/atoms/Badge/Badge";
import { Link } from "@/components/atoms/Link/Link";
import { Text } from "@/components/atoms/Text/Text";
import {
  formatBlockedAddress,
  formatDate,
  getAdminUsersHref,
  getSubscriptionLabel,
} from "../../../helpers";

// More than this many shared IPs collapse into "+N IP".
const SHOWN_IPS = 2;

interface Props {
  account: RelatedAccount;
}

export function RelatedAccountRow({ account }: Props) {
  const ips = account.via_ips.map(formatBlockedAddress);
  const hidden = ips.length - SHOWN_IPS;

  return (
    <li className="grid grid-cols-1 gap-x-4 gap-y-1 py-2.5 first:pt-0 last:pb-0 md:grid-cols-[minmax(0,1fr)_minmax(0,auto)_7rem_4.5rem] md:items-center">
      <div className="flex min-w-0 flex-col">
        <Link
          href={getAdminUsersHref(account.id)}
          variant="secondary"
          className="truncate"
        >
          <span translate="no">{account.email}</span>
        </Link>
        {account.name.trim() ? (
          <Text variant="small" className="text-foreground-tertiary">
            {account.name}
          </Text>
        ) : null}
        <Text
          variant="small"
          className={
            account.payment_status === "past_due"
              ? "text-warning-foreground"
              : "text-foreground-tertiary"
          }
        >
          {getSubscriptionLabel(account.plan, account.payment_status)}
        </Text>
      </div>
      <div className="flex flex-wrap gap-1">
        {ips.slice(0, SHOWN_IPS).map((ip) => (
          <Badge key={ip} variant="info" size="small">
            IP {ip}
          </Badge>
        ))}
        {hidden > 0 ? (
          <Badge variant="info" size="small">
            +{hidden} IP
          </Badge>
        ) : null}
        {account.via_domain ? (
          <Badge variant="info" size="small">
            Domain
          </Badge>
        ) : null}
        {account.has_profile ? null : (
          <Badge variant="info" size="small">
            No profile
          </Badge>
        )}
      </div>
      <Text variant="small" as="span" className="text-foreground-tertiary">
        {formatDate(account.created_at)}
      </Text>
      <span className="md:text-right">
        {account.banned ? (
          <Badge variant="error" size="small">
            Banned
          </Badge>
        ) : null}
      </span>
    </li>
  );
}
