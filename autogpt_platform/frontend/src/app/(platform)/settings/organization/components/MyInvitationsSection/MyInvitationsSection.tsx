"use client";

import { Badge } from "@/components/atoms/Badge/Badge";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";

import { useMyInvitationsSection } from "./useMyInvitationsSection";

export function MyInvitationsSection() {
  const { invitations, isAccepting, isDeclining, handleAccept, handleDecline } =
    useMyInvitationsSection();

  if (invitations.length === 0) {
    return null;
  }

  return (
    <section
      className="flex flex-col gap-3 rounded-xl border border-violet-200 bg-violet-50 p-4"
      data-testid="my-invitations-section"
    >
      <Text variant="h4" as="h2">
        You&apos;ve been invited
      </Text>
      <ul className="flex flex-col gap-2">
        {invitations.map((invitation) => (
          <li
            key={invitation.id}
            className="flex items-center gap-3"
            data-testid="my-invitation-row"
          >
            <div className="flex min-w-0 flex-1 flex-col">
              <span className="truncate text-sm font-medium">
                {invitation.org_name}
              </span>
              <span className="text-xs text-zinc-500">
                Expires {new Date(invitation.expires_at).toLocaleDateString()}
              </span>
            </div>
            {invitation.is_admin ? <Badge variant="info">Admin</Badge> : null}
            <Button
              size="small"
              loading={isAccepting}
              onClick={() => handleAccept(invitation)}
            >
              Accept
            </Button>
            <Button
              variant="secondary"
              size="small"
              loading={isDeclining}
              onClick={() => handleDecline(invitation)}
            >
              Decline
            </Button>
          </li>
        ))}
      </ul>
    </section>
  );
}
