"use client";

import type { AccountIPReport } from "@/app/api/__generated__/models/accountIPReport";
import { Badge } from "@/components/atoms/Badge/Badge";
import { Text } from "@/components/atoms/Text/Text";
import { pluralize } from "../../../helpers";
import { BlockDomainDialog } from "../../AccessDialogs/BlockDomainDialog/BlockDomainDialog";
import { UnblockDialog } from "../../AccessDialogs/UnblockDialog";

interface Props {
  report: AccountIPReport;
  signedInUserId: string | undefined;
}

export function DomainLine({ report, signedInUserId }: Props) {
  const { domain } = report;
  const name = <span className="font-medium">{domain.value}</span>;

  if (domain.is_public_provider) {
    return (
      <Text variant="body" className="text-zinc-700">
        Email domain {name} is a public mail provider, so its accounts
        aren&apos;t listed and it can&apos;t be blocked.
      </Text>
    );
  }
  return (
    <div className="flex flex-wrap items-center justify-between gap-2">
      <Text variant="body" className="text-zinc-700">
        Email domain {name} ·{" "}
        {domain.other_account_count > 0
          ? pluralize(domain.other_account_count, "other account")
          : "no other accounts"}
      </Text>
      <div className="flex items-center gap-2">
        {domain.block ? (
          <>
            <Badge variant="warning" size="small">
              Blocked
            </Badge>
            <UnblockDialog block={domain.block} triggerVariant="link" />
          </>
        ) : null}
        <BlockDomainDialog
          report={report}
          signedInUserId={signedInUserId}
          showTrigger={!domain.block}
        />
      </div>
    </div>
  );
}
