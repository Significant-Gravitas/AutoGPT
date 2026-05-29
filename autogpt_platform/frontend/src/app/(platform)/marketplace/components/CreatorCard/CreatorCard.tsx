"use client";

import Avatar, {
  AvatarFallback,
  AvatarImage,
} from "@/components/atoms/Avatar/Avatar";
import { Text } from "@/components/atoms/Text/Text";
import { backgroundColor } from "./helper";

interface Props {
  creatorName: string;
  creatorImage: string | null;
  bio: string;
  agentsUploaded: number;
  onClick: () => void;
  index: number;
}

export function CreatorCard({
  creatorName,
  creatorImage,
  bio,
  agentsUploaded,
  onClick,
  index,
}: Props) {
  return (
    <button
      type="button"
      className={`relative flex h-[16rem] w-full cursor-pointer flex-col items-start rounded-2xl border p-4 text-left shadow-md transition-all duration-300 hover:shadow-lg ${backgroundColor(index)}`}
      onClick={onClick}
      data-testid="creator-card"
    >
      {/* Avatar */}
      <Avatar className="h-14 w-14 shrink-0">
        {creatorImage && (
          <AvatarImage src={creatorImage} alt={`${creatorName} avatar`} />
        )}
        <AvatarFallback size={56}>{creatorName.charAt(0)}</AvatarFallback>
      </Avatar>

      <div className="mt-3 flex w-full flex-1 flex-col">
        <Text variant="h4" className="leading-tight">
          {creatorName}
        </Text>
        <div className="mt-2 flex w-full flex-col">
          <Text variant="body" className="line-clamp-3 leading-normal">
            {bio}
          </Text>
        </div>
      </div>

      {/* Stats */}
      <Text variant="body" className="absolute bottom-4 left-4 text-zinc-500">
        {agentsUploaded} {agentsUploaded === 1 ? "agent" : "agents"}
      </Text>
    </button>
  );
}
