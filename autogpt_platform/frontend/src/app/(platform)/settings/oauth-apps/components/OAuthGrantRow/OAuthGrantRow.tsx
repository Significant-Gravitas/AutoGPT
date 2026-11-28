"use client";

import type { OAuthGrantInfo } from "@/app/api/__generated__/models/oAuthGrantInfo";
import Avatar, {
  AvatarFallback,
  AvatarImage,
} from "@/components/atoms/Avatar/Avatar";
import { Button } from "@/components/atoms/Button/Button";
import { RelativeTime } from "@/components/atoms/RelativeTime/RelativeTime";
import { Text } from "@/components/atoms/Text/Text";
import { SCOPE_DESCRIPTIONS } from "@/lib/api-key-scopes";

interface Props {
  grant: OAuthGrantInfo;
  onRevoke: () => void;
}

export function OAuthGrantRow({ grant, onRevoke }: Props) {
  return (
    <li className="flex flex-col gap-4 px-5 py-4 sm:flex-row sm:items-start">
      <div className="flex min-w-0 flex-1 items-start gap-4">
        <Avatar className="h-10 w-10">
          {grant.logo_url && (
            <AvatarImage as="img" src={grant.logo_url} alt="" />
          )}
          <AvatarFallback>{grant.name}</AvatarFallback>
        </Avatar>
        <div className="flex min-w-0 flex-col gap-1">
          <Text variant="body-medium" as="h2" className="truncate">
            {grant.name}
          </Text>
          <Text variant="small-loose" as="span" tone="muted">
            Last active <RelativeTime date={grant.last_active_at} />
          </Text>
          <ul
            aria-label={`What ${grant.name} can do`}
            className="mt-2 flex flex-col gap-1"
          >
            {grant.scopes.map((scope) => (
              <li key={scope}>
                <Text variant="small" as="span" tone="body">
                  {SCOPE_DESCRIPTIONS[scope] ?? scope}
                </Text>
              </li>
            ))}
          </ul>
        </div>
      </div>
      <Button
        variant="secondary"
        size="small"
        onClick={onRevoke}
        aria-label={`Revoke access for ${grant.name}`}
        className="self-start"
      >
        Revoke
      </Button>
    </li>
  );
}
