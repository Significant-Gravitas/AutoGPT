"use client";

import {
  Avatar,
  AvatarFallback,
  AvatarImage,
} from "@/components/atoms/Avatar/Avatar";
import { Badge } from "@/components/atoms/Badge/Badge";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { UserSwitchIcon } from "@hugeicons/core-free-icons";
import { AdminSection } from "../../../components/AdminSection";
import { useAdminImpersonation } from "../../../components/useAdminImpersonation";
import { type AdminUser, getDisplayName } from "../../helpers";
import { BanDialog } from "./components/BanDialog";
import { ImpersonateDialog } from "./components/ImpersonateDialog";
import { UnbanDialog } from "./components/UnbanDialog";

interface Props {
  user: AdminUser;
  isSelf: boolean;
  canAct: boolean;
}

export function UserHeader({ user, isSelf, canAct }: Props) {
  const impersonation = useAdminImpersonation();
  const displayName = getDisplayName(user);
  const isImpersonatingThisUser = impersonation.impersonatedUserId === user.id;

  return (
    <AdminSection className="flex flex-wrap items-start justify-between gap-4">
      <div className="flex min-w-0 items-center gap-3">
        <Avatar className="h-10 w-10 shrink-0">
          {user.image ? <AvatarImage src={user.image} alt="" /> : null}
          <AvatarFallback accessibleLabel={displayName} className="bg-zinc-100">
            <span className="text-base font-medium text-zinc-700">
              {displayName.charAt(0).toUpperCase()}
            </span>
          </AvatarFallback>
        </Avatar>
        <div className="flex min-w-0 flex-col">
          <div className="flex flex-wrap items-center gap-2">
            <Text variant="large-medium" as="h2" className="break-all">
              {displayName}
            </Text>
            {isSelf ? (
              <Badge variant="info" size="small">
                You
              </Badge>
            ) : null}
            {user.role === "admin" ? (
              <Badge variant="info" size="small">
                Admin
              </Badge>
            ) : null}
            {user.banned ? (
              <Badge variant="error" size="small">
                Banned
              </Badge>
            ) : null}
            {user.emailVerified ? null : (
              <Badge variant="warning" size="small">
                Unverified
              </Badge>
            )}
          </div>
          <Text
            variant="body"
            className="break-all text-zinc-600"
            translate="no"
          >
            {user.email}
          </Text>
        </div>
      </div>

      {canAct ? (
        <div className="flex flex-wrap gap-2">
          {isImpersonatingThisUser ? (
            <Button
              variant="secondary"
              size="small"
              leadingIcon={UserSwitchIcon}
              onClick={impersonation.stopImpersonating}
            >
              Stop impersonating
            </Button>
          ) : (
            <ImpersonateDialog
              user={user}
              currentImpersonationID={impersonation.impersonatedUserId}
              isStarting={impersonation.isStarting}
              onConfirm={() => impersonation.startImpersonating(user.id)}
            />
          )}
          {user.banned ? (
            <UnbanDialog user={user} />
          ) : (
            <BanDialog user={user} />
          )}
        </div>
      ) : null}
    </AdminSection>
  );
}
