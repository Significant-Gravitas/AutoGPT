"use client";

import type { OrgResponse } from "@/app/api/__generated__/models/orgResponse";
import { Avatar, AvatarImage } from "@/components/atoms/Avatar/Avatar";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";

import { getOrgInitials } from "./helpers";
import { useOrgAvatarControl } from "./useOrgAvatarControl";

interface Props {
  org: OrgResponse;
  isAdmin: boolean;
  onSaved: () => void;
}

export function OrgAvatarControl({ org, isAdmin, onSaved }: Props) {
  const { fileRef, isPending, openFilePicker, handleFileChange } =
    useOrgAvatarControl({ org, onSaved });

  return (
    <div className="flex items-center gap-4" data-testid="org-avatar-control">
      <Avatar className="h-16 w-16 bg-zinc-100">
        {org.avatar_url ? (
          <AvatarImage
            src={org.avatar_url}
            alt={org.name}
            width={64}
            height={64}
          />
        ) : (
          <span
            className="flex h-full w-full items-center justify-center text-lg font-medium text-zinc-600"
            data-testid="org-avatar-initials"
          >
            {getOrgInitials(org.name)}
          </span>
        )}
      </Avatar>

      {isAdmin ? (
        <div className="flex flex-col gap-1">
          <Button
            type="button"
            variant="secondary"
            size="small"
            loading={isPending}
            onClick={openFilePicker}
          >
            Change
          </Button>
          <Text variant="small" className="text-zinc-500">
            PNG, JPG, GIF or WebP.
          </Text>
        </div>
      ) : null}

      <input
        ref={fileRef}
        type="file"
        accept="image/png,image/jpeg,image/webp,image/gif"
        className="hidden"
        aria-label="Upload organization avatar"
        onChange={handleFileChange}
      />
    </div>
  );
}
