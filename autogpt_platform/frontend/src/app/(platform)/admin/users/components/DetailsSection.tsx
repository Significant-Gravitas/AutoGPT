"use client";

import { useGetV2GetUserRateLimit } from "@/app/api/__generated__/endpoints/admin/admin";
import { okData } from "@/app/api/helpers";
import { Badge } from "@/components/atoms/Badge/Badge";
import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { Copy01Icon } from "@hugeicons/core-free-icons";
import { ReactNode } from "react";
import { AdminSection } from "../../components/AdminSection";
import { type AdminUser, formatDate, getPlanName } from "../helpers";

interface Props {
  user: AdminUser;
}

export function DetailsSection({ user }: Props) {
  const { toast } = useToast();
  const { data: rateLimit, isError: planFailed } = useGetV2GetUserRateLimit(
    { user_id: user.id },
    { query: { select: okData } },
  );

  async function copyUserID() {
    try {
      await navigator.clipboard.writeText(user.id);
      toast({ title: "User ID copied" });
    } catch {
      toast({
        title: "Couldn't copy the user ID",
        description: "Select it and copy it by hand.",
        variant: "destructive",
      });
    }
  }

  return (
    <AdminSection title="Details">
      <dl className="flex flex-col gap-3">
        <Row label="User ID">
          <span className="flex items-start gap-1">
            <span
              className="min-w-0 break-all font-mono text-[13px]"
              translate="no"
            >
              {user.id}
            </span>
            <Button
              variant="icon"
              size="icon-xs"
              aria-label="Copy user ID"
              onClick={() => void copyUserID()}
            >
              <Icon icon={Copy01Icon} size={14} aria-hidden />
            </Button>
          </span>
        </Row>
        <Row label="Email">
          <span className="flex flex-wrap items-center gap-1.5">
            <span className="break-all" translate="no">
              {user.email}
            </span>
            {user.emailVerified ? (
              <Badge variant="success" size="small">
                Verified
              </Badge>
            ) : (
              <Badge variant="warning" size="small">
                Unverified
              </Badge>
            )}
          </span>
        </Row>
        <Row label="Role">{user.role === "admin" ? "Admin" : "User"}</Row>
        <Row label="Plan">
          {rateLimit
            ? getPlanName(rateLimit.tier)
            : planFailed
              ? "Couldn't load"
              : "—"}
        </Row>
        <Row label="Joined">{formatDate(user.createdAt)}</Row>
        <Row label="Status">
          {user.banned ? (
            <span className="text-red-700">
              {user.banExpires
                ? `Banned until ${formatDate(user.banExpires)}`
                : "Banned permanently"}
            </span>
          ) : (
            "Active"
          )}
        </Row>
      </dl>
    </AdminSection>
  );
}

function Row({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="flex gap-3">
      <Text variant="small" as="dt" className="w-[88px] shrink-0 text-zinc-500">
        {label}
      </Text>
      <Text variant="body" as="dd" className="min-w-0 flex-1">
        {children}
      </Text>
    </div>
  );
}
