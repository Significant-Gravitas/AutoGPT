"use client";

import Avatar, {
  AvatarFallback,
  AvatarImage,
} from "@/components/atoms/Avatar/Avatar";
import { Text } from "@/components/atoms/Text/Text";
import { backgroundColor } from "./helper";

interface Props {
  creatorName: string;
  creatorUsername?: string;
  creatorImage: string | null;
  bio: string;
  agentsUploaded: number;
  onClick: () => void;
  index: number;
}

export function CreatorCard({
  creatorName,
  creatorUsername,
  creatorImage,
  bio,
  agentsUploaded,
  onClick,
  index,
}: Props) {
  return (
    <button
      type="button"
      // oxlint-disable-next-line design-system/no-arbitrary-values -- the card's hover lift shadow has no shadow token
      className={`relative flex h-[16rem] w-full cursor-pointer flex-col items-start rounded-2xl border p-4 text-left shadow-message-table transition-all duration-200 hover:-translate-y-0.5 hover:shadow-[0_16px_40px_-16px_rgba(16,24,40,0.18)] ${backgroundColor(index)}`}
      onClick={onClick}
      data-testid="creator-card"
    >
      {/* Avatar */}
      <Avatar className="h-14 w-14 shrink-0">
        {creatorImage && (
          <AvatarImage src={creatorImage} alt={`${creatorName} avatar`} />
        )}
        <AvatarFallback size={56}>
          {creatorUsername || creatorName}
        </AvatarFallback>
      </Avatar>

      <div className="mt-3 flex w-full flex-1 flex-col">
        <Text variant="h4">{creatorName}</Text>
        <div className="mt-2 flex w-full flex-col">
          <Text variant="body" className="line-clamp-3">
            {bio}
          </Text>
        </div>
      </div>

      {/* Stats */}
      <Text variant="body" tone="muted" className="absolute bottom-4 left-4">
        {agentsUploaded} {agentsUploaded === 1 ? "workflow" : "workflows"}
      </Text>
    </button>
  );
}
