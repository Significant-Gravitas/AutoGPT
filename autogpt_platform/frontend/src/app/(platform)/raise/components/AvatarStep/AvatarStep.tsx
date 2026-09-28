"use client";

import { cn } from "@/lib/utils";
import { ExpertAvatar } from "@/components/molecules/ExpertAvatar/ExpertAvatar";
import { bubbleClassFor } from "../ColorStep/helpers";
import { ExpertAvatarPicker } from "@/components/molecules/ExpertAvatarPicker/ExpertAvatarPicker";

interface Props {
  name: string;
  color: string | null;
  avatarUrl: string | null;
  onPick: (avatarUrl: string, colorId: string) => void;
}

export function AvatarStep({ name, color, avatarUrl, onPick }: Props) {
  if (avatarUrl) {
    return (
      <div
        className={cn(
          "ml-auto flex w-fit items-center gap-3 rounded-full border py-2 pl-2 pr-5",
          bubbleClassFor(color) ?? "border-accent bg-accent/5",
        )}
      >
        <ExpertAvatar
          name={name || "Your expert"}
          avatarUrl={avatarUrl}
          size={40}
        />
        <span className="text-sm font-medium text-foreground">
          {name ? `${name} has an avatar` : "Picture set"}
        </span>
      </div>
    );
  }

  return <ExpertAvatarPicker name={name} color={color} onPick={onPick} />;
}
