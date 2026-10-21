"use client";

import { Text } from "@/components/atoms/Text/Text";
import { AdminSection } from "../../../components/AdminSection";
import { useIPReport, useOwnSessionIP } from "../../useIPReport";
import { ReportError, ReportLoading } from "../ReportStates";
import { IPRow } from "./components/IPRow";

interface Props {
  userId: string;
}

export function SignInsSection({ userId }: Props) {
  const { report, isLoading, isForbidden, refetch } = useIPReport(userId);
  const ownIP = useOwnSessionIP();

  return (
    <AdminSection
      title="Sign-ins"
      description={
        report
          ? `Where this account signed up and signed in. Kept ${report.retention_days} days.`
          : "Where this account signed up and signed in."
      }
    >
      {isLoading ? (
        <ReportLoading />
      ) : !report ? (
        <ReportError
          message="Couldn't load sign-ins."
          isForbidden={isForbidden}
          onRetry={() => void refetch()}
        />
      ) : report.ips.length === 0 ? (
        <Text variant="small" className="text-zinc-500">
          No sign-ins on record. The log keeps {report.retention_days} days and
          started recently; accounts from before it show nothing until they sign
          in again.
        </Text>
      ) : (
        <div className="flex flex-col gap-2">
          {report.ips.some((ip) => ip.has_sign_up) ? null : (
            <Text variant="small" className="text-zinc-500">
              Sign-up address unknown: the account predates the log.
            </Text>
          )}
          <ul className="divide-y divide-zinc-200">
            {report.ips.map((ip) => (
              <IPRow key={ip.ip_address} ip={ip} ownIP={ownIP} />
            ))}
          </ul>
        </div>
      )}
    </AdminSection>
  );
}
