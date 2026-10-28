"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { useAuth } from "@/lib/auth/hooks/useAuth";
import { useState } from "react";
import { AdminSection } from "../../../components/AdminSection";
import type { AdminUser } from "../../helpers";
import { useIPReport, useOwnSessionIP } from "../../useIPReport";
import { BanAllDialog } from "../AccessDialogs/BanAllDialog/BanAllDialog";
import { ReportError, ReportLoading } from "../ReportStates";
import { DomainLine } from "./components/DomainLine";
import { RelatedAccountRow } from "./components/RelatedAccountRow";

const SHOWN_AT_FIRST = 10;

interface Props {
  user: AdminUser;
}

export function RelatedAccountsSection({ user }: Props) {
  const { report, isLoading, isForbidden, refetch } = useIPReport(user.id);
  const { user: signedInUser } = useAuth();
  const ownIP = useOwnSessionIP();
  const [showAll, setShowAll] = useState(false);

  return (
    <AdminSection
      title="Related accounts"
      description="Other accounts that signed in from the same addresses or use the same email domain."
    >
      {isLoading ? (
        <ReportLoading />
      ) : !report ? (
        <ReportError
          message="Couldn't load related accounts."
          isForbidden={isForbidden}
          onRetry={() => void refetch()}
        />
      ) : (
        <div className="flex flex-col gap-3">
          <DomainLine report={report} signedInUserId={signedInUser?.id} />
          {report.related.length === 0 ? (
            <Text variant="small" className="text-foreground-tertiary">
              {report.ips.length === 0
                ? "No addresses on record, so related accounts are matched by email domain only: none."
                : "No other accounts share this account's addresses or email domain."}
            </Text>
          ) : (
            <>
              <ul className="divide-y divide-border-base border-t border-border-base pt-2.5">
                {(showAll
                  ? report.related
                  : report.related.slice(0, SHOWN_AT_FIRST)
                ).map((account) => (
                  <RelatedAccountRow key={account.id} account={account} />
                ))}
              </ul>
              {!showAll && report.related.length > SHOWN_AT_FIRST ? (
                <div className="flex justify-center">
                  <Button
                    variant="ghost"
                    size="small"
                    onClick={() => setShowAll(true)}
                  >
                    Show {report.related.length - SHOWN_AT_FIRST} more
                  </Button>
                </div>
              ) : null}
              <div className="flex flex-col items-end gap-2 border-t border-border-base pt-3">
                {report.related_total > report.related.length ? (
                  <Text variant="small" className="text-foreground-tertiary">
                    Showing {report.related.length.toLocaleString("en-GB")} of{" "}
                    {report.related_total.toLocaleString("en-GB")}. Ban all
                    covers these {report.related.length.toLocaleString("en-GB")}
                    .
                  </Text>
                ) : null}
                <BanAllDialog
                  user={user}
                  report={report}
                  signedInUserId={signedInUser?.id}
                  ownIP={ownIP}
                />
              </div>
            </>
          )}
        </div>
      )}
    </AdminSection>
  );
}
