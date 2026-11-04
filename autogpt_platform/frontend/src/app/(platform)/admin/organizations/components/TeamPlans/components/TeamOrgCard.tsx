"use client";

import type { OrgPlanSummary } from "@/app/api/__generated__/models/orgPlanSummary";
import { Badge } from "@/components/atoms/Badge/Badge";
import { Text } from "@/components/atoms/Text/Text";
import type { ReactNode } from "react";
import { getPlanName } from "../../../../users/helpers";
import {
  describeEnd,
  describeNotInEffect,
  describeSeatsInUse,
  formatBalance,
  getStatusLabel,
  isOverSeats,
} from "../helpers";
import { AddCreditsDialog } from "./AddCreditsDialog/AddCreditsDialog";
import { SetPlanDialog } from "./SetPlanDialog/SetPlanDialog";

interface Props {
  org: OrgPlanSummary;
}

export function TeamOrgCard({ org }: Props) {
  const { plan } = org;
  const notInEffect = describeNotInEffect(plan);

  return (
    <li
      data-testid="team-org"
      className="flex flex-col gap-3 py-4 first:pt-0 last:pb-0"
    >
      <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
        <div className="flex min-w-0 flex-col gap-1">
          <div className="flex flex-wrap items-center gap-x-2 gap-y-1">
            <Text variant="body-medium" as="h3" className="truncate">
              {org.name}
            </Text>
            <Text variant="small-mono" as="span" tone="muted" translate="no">
              {org.slug}
            </Text>
          </div>
          {org.owner_email ? (
            <Text variant="small" tone="muted">
              {`Owner: ${org.owner_email}`}
            </Text>
          ) : null}
        </div>
        <div className="flex shrink-0 flex-wrap gap-2">
          <SetPlanDialog org={org} />
          <AddCreditsDialog org={org} />
        </div>
      </div>
      <div className="flex flex-col items-start gap-1.5">
        <Badge variant={notInEffect ? "warning" : "success"} size="small">
          {notInEffect ? "Not in effect" : "In effect"}
        </Badge>
        {notInEffect ? (
          <Text variant="small" tone="secondary">
            {notInEffect}
          </Text>
        ) : null}
      </div>
      <dl className="grid grid-cols-1 gap-y-1 sm:grid-cols-[8rem_1fr] sm:gap-x-4 sm:gap-y-2">
        <Fact label="Plan">{plan ? getPlanName(plan.tier) : "No plan"}</Fact>
        {plan ? (
          <Fact label="Status">{getStatusLabel(plan.status)}</Fact>
        ) : null}
        {plan ? <Fact label="End date">{describeEnd(plan)}</Fact> : null}
        <Fact label="Seats" isWarning={isOverSeats(org)}>
          {describeSeatsInUse(org)}
        </Fact>
        <Fact label="Balance">{formatBalance(org.balance)}</Fact>
        {plan?.note ? <Fact label="Note">{plan.note}</Fact> : null}
      </dl>
    </li>
  );
}

interface FactProps {
  label: string;
  isWarning?: boolean;
  children: ReactNode;
}

function Fact({ label, isWarning = false, children }: FactProps) {
  return (
    <>
      <dt>
        <Text variant="small" as="span" tone="muted">
          {label}
        </Text>
      </dt>
      <dd className="min-w-0">
        <Text
          variant="small"
          as="span"
          tone={isWarning ? "warning" : undefined}
          className="break-words"
        >
          {children}
        </Text>
      </dd>
    </>
  );
}
