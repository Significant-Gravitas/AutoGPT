"use client";

import { useAuth } from "@/lib/auth/hooks/useAuth";
import type { AdminUser } from "../../helpers";
import { AutoPilotUsageSection } from "../AutoPilotUsageSection/AutoPilotUsageSection";
import { BannedAlert } from "../BannedAlert";
import { DetailsSection } from "../DetailsSection";
import { OrganisationsSection } from "../OrganisationsSection";
import { RelatedAccountsSection } from "../RelatedAccountsSection/RelatedAccountsSection";
import { SignInsSection } from "../SignInsSection/SignInsSection";
import { SpendingSection } from "../SpendingSection";
import { UserHeader } from "../UserHeader/UserHeader";

interface Props {
  user: AdminUser;
}

// The column wrappers are `display: contents` below lg, so the `order-*`
// classes interleave both columns into one: identity before controls.
export function UserView({ user }: Props) {
  const { user: signedInUser } = useAuth();
  const isSelf = signedInUser?.id === user.id;
  // Until the signed-in admin is known, their own row would offer Ban.
  const canAct = Boolean(signedInUser) && !isSelf;

  return (
    <div className="flex flex-col gap-6">
      <UserHeader user={user} isSelf={isSelf} canAct={canAct} />
      {user.banned ? <BannedAlert user={user} /> : null}
      <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_320px]">
        <div className="contents lg:flex lg:flex-col lg:gap-6">
          <div className="order-2 lg:order-none">
            <AutoPilotUsageSection userId={user.id} />
          </div>
          <div className="order-3 lg:order-none">
            <OrganisationsSection userId={user.id} />
          </div>
          <div className="order-4 lg:order-none">
            <SignInsSection userId={user.id} />
          </div>
          <div className="order-5 lg:order-none">
            <RelatedAccountsSection user={user} />
          </div>
        </div>
        <div className="contents lg:flex lg:flex-col lg:gap-6">
          <div className="order-1 lg:order-none">
            <DetailsSection user={user} />
          </div>
          <div className="order-6 lg:order-none">
            <SpendingSection userId={user.id} />
          </div>
        </div>
      </div>
    </div>
  );
}
