"use client";

import { Button } from "@/components/atoms/Button/Button";
import { cn } from "@/lib/utils";
import Image from "next/image";
import { bubbleClassFor } from "../ColorStep/helpers";
import { ACCEPTED_AVATAR_TYPES } from "./helpers";
import { useAvatarStep } from "./useAvatarStep";

interface Props {
  name: string;
  color: string | null;
  avatarUrl: string | null;
  isSkipped: boolean;
  onPick: (avatarUrl: string) => void;
  onSkip: () => void;
}

export function AvatarStep({
  name,
  color,
  avatarUrl,
  isSkipped,
  onPick,
  onSkip,
}: Props) {
  const { fileInputRef, isUploading, openFilePicker, handleFileChange } =
    useAvatarStep({ onPick });

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

  if (isSkipped) {
    return (
      <div
        className={cn(
          "ml-auto w-fit rounded-full border px-5 py-2.5 text-sm font-medium text-foreground",
          bubbleClassFor(color) ?? "border-accent bg-accent/5",
        )}
      >
        No picture for now
      </div>
    );
  }

  return (
    <div className="flex flex-wrap justify-end gap-2.5">
      <input
        ref={fileInputRef}
        type="file"
        accept={ACCEPTED_AVATAR_TYPES}
        className="sr-only"
        tabIndex={-1}
        aria-hidden
        onChange={(event) => handleFileChange(event.target.files?.[0])}
      />
      <Button
        variant="secondary"
        size="small"
        className="rounded-full"
        onClick={openFilePicker}
        loading={isUploading}
      >
        Upload a picture
      </Button>
      <Button
        variant="ghost"
        size="small"
        className="rounded-full"
        onClick={onSkip}
        disabled={isUploading}
      >
        Skip
      </Button>
    </div>
  );
}
