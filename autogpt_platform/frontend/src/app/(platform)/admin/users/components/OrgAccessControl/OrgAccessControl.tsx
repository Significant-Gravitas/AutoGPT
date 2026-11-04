"use client";

import type { OrgAccessStatus } from "@/app/api/__generated__/models/orgAccessStatus";
import { Badge } from "@/components/atoms/Badge/Badge";
import { Button } from "@/components/atoms/Button/Button";
import { Link } from "@/components/atoms/Link/Link";
import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { Text } from "@/components/atoms/Text/Text";
import { DisableAccessDialog } from "../../../organizations/components/DisableAccessDialog/DisableAccessDialog";
import {
  formatGrantDate,
  getTeamPlansHref,
} from "../../../organizations/helpers";
import { useOrgAccessControl } from "./useOrgAccessControl";

interface Props {
  userId: string;
}

export function OrgAccessControl({ userId }: Props) {
  const { status, isLoading, retry, enable, disable, isSaving } =
    useOrgAccessControl(userId);

  if (isLoading) return <Skeleton className="h-10 w-full rounded-lg" />;
  if (!status) {
    return (
      <div className="flex items-center justify-between gap-4">
        <Text variant="small" tone="muted">
          Couldn&apos;t load organization access.
        </Text>
        <Button variant="secondary" size="small" onClick={retry}>
          Retry
        </Button>
      </div>
    );
  }

  return (
    <div
      className="flex items-center justify-between gap-3"
      data-testid="org-access-control"
    >
      <div className="flex min-w-0 flex-col gap-0.5">
        <div className="flex items-center gap-2">
          <Text variant="body-medium" as="span">
            Organization features
          </Text>
          <Badge variant={status.has_access ? "success" : "info"} size="small">
            {status.has_access ? "On" : "Off"}
          </Badge>
        </div>
        <Text variant="small" tone="muted">
          {describeAccess(status)}
        </Text>
        <Link
          href={getTeamPlansHref(status.email)}
          variant="secondary"
          className="w-fit"
        >
          Team plans
        </Link>
      </div>
      {status.grant ? (
        <DisableAccessDialog
          email={status.email}
          isDisabling={isSaving}
          onConfirm={disable}
        />
      ) : (
        <Button
          variant="secondary"
          size="small"
          loading={isSaving}
          onClick={() => void enable()}
        >
          Turn on
        </Button>
      )}
    </div>
  );
}

function describeAccess(status: OrgAccessStatus) {
  if (status.grant) {
    const by = status.grant.granted_by_email
      ? ` by ${status.grant.granted_by_email}`
      : "";
    return `Turned on ${formatGrantDate(status.grant.granted_at)}${by}.`;
  }
  const [inherited] = status.inherited_from;
  if (inherited) {
    return inherited.via === "invitation"
      ? `Invited to ${inherited.org_name}, which has organizations.`
      : `Member of ${inherited.org_name}, which has organizations.`;
  }
  return "Organizations are hidden for this account.";
}
