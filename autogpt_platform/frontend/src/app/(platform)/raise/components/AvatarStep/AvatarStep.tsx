"use client";

import { cn } from "@/lib/utils";
import Image from "next/image";
import { bubbleClassFor } from "../ColorStep/helpers";
import { NotionAvatarPicker } from "./components/NotionAvatarPicker/NotionAvatarPicker";

interface Props {
  name: string;
  color: string | null;
  avatarUrl: string | null;
  onPick: (avatarUrl: string, colorId: string) => void;
}

// The face and the colour are one answer, and the step opens straight onto the
// picker: there is no version of an expert without a face.
export function AvatarStep({ name, color, avatarUrl, onPick }: Props) {
  if (avatarUrl) {
    return (
      <div
        className={cn(
          "ml-auto flex w-fit items-center gap-3 rounded-full border py-2 pl-2 pr-5",
          bubbleClassFor(color) ?? "border-accent bg-accent/5",
        )}
      >
        <Image
          src={avatarUrl}
          alt={`${name || "Your expert"}'s picture`}
          width={40}
          height={40}
          className="size-10 rounded-full object-cover"
          unoptimized
        />
        <span className="text-sm font-medium text-foreground">
          {name ? `${name} has a face` : "Picture set"}
        </span>
      </div>
    );
  }

  return <NotionAvatarPicker name={name} color={color} onPick={onPick} />;
}
