"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { Text } from "@/components/atoms/Text/Text";
import { pluralize } from "../../../users/helpers";
import { getErrorMessage } from "../../helpers";
import { AccountLookupForm } from "./components/AccountLookupForm";
import { TeamOrgCard } from "./components/TeamOrgCard";
import { useTeamPlans } from "./useTeamPlans";

export function TeamPlans() {
  const plans = useTeamPlans();

  return (
    <div className="flex flex-col gap-4">
      <AccountLookupForm
        initialEmail={plans.email ?? ""}
        isLoading={plans.isLoading}
        onLookUp={plans.lookUp}
      />
      <TeamOrgResults {...plans} />
    </div>
  );
}

type ResultsProps = ReturnType<typeof useTeamPlans>;

function TeamOrgResults({
  email,
  orgs,
  isLoading,
  isNotFound,
  error,
  retry,
}: ResultsProps) {
  if (!email) return null;
  if (isLoading) {
    return (
      <div className="flex flex-col gap-2" aria-busy="true">
        <Skeleton className="h-24 w-full rounded-lg" />
        <Skeleton className="h-24 w-full rounded-lg" />
      </div>
    );
  }
  if (isNotFound) {
    return (
      <Text variant="small" tone="muted">
        {`No account signs in with ${email}.`}
      </Text>
    );
  }
  if (error || !orgs) {
    return (
      <div className="flex items-center justify-between gap-4">
        <Text variant="small" tone="muted">
          {`Couldn't load the team organizations of ${email}. ${getErrorMessage(error)}`}
        </Text>
        <Button variant="secondary" size="small" onClick={retry}>
          Retry
        </Button>
      </div>
    );
  }
  if (orgs.length === 0) {
    return (
      <Text variant="small" tone="muted">
        {`${email} isn't in any team organization.`}
      </Text>
    );
  }
  return (
    <div className="flex flex-col gap-3">
      <Text variant="small" tone="muted">
        {`${email} is in ${pluralize(orgs.length, "team organization")}.`}
      </Text>
      <ul className="divide-y divide-border-base">
        {orgs.map((org) => (
          <TeamOrgCard key={org.org_id} org={org} />
        ))}
      </ul>
    </div>
  );
}
