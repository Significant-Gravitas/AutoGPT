"use client";

import { Expert } from "@/app/api/__generated__/models/expert";
import {
  Avatar,
  AvatarFallback,
  AvatarImage,
} from "@/components/atoms/Avatar/Avatar";
import { Icon } from "@/components/atoms/Icon/Icon";
import { BotAvatar } from "@/components/molecules/BotAvatar/BotAvatar";
import {
  expertAvatarConfig,
  isUploadedAvatar,
} from "@/components/molecules/BotAvatar/helpers";
import { Camera01Icon, Loading03Icon } from "@hugeicons/core-free-icons";
import { ChangeEvent, useRef } from "react";
import { useExpertAvatarButton } from "./useExpertAvatarButton";

interface Props {
  expert: Expert;
}

export function ExpertAvatarButton({ expert }: Props) {
  const fileRef = useRef<HTMLInputElement>(null);
  const { uploadAvatar, isUploading } = useExpertAvatarButton(expert.id);

  function openFilePicker() {
    if (isUploading) return;
    fileRef.current?.click();
  }

  async function handleChange(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (file) await uploadAvatar(file);
  }

  return (
    <>
      <button
        type="button"
        onClick={openFilePicker}
        disabled={isUploading}
        aria-label={`Change ${expert.name}'s photo`}
        className="group relative size-24 shrink-0 cursor-pointer rounded-full outline-none transition-transform duration-150 ease-out focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 active:scale-[0.97] disabled:cursor-wait"
      >
        {isUploadedAvatar(expert.avatar_url) ? (
          <Avatar className="size-24 bg-background ring-4 ring-background">
            <AvatarImage
              src={expert.avatar_url ?? undefined}
              alt={expert.name}
              width={192}
              height={192}
            />
            <AvatarFallback>{expert.name}</AvatarFallback>
          </Avatar>
        ) : (
          <span className="flex size-24 items-center justify-center rounded-full border border-black bg-background ring-4 ring-background">
            <BotAvatar
              config={expertAvatarConfig({
                name: expert.name,
                avatarUrl: expert.avatar_url,
                color: expert.color,
              })}
              size={84}
              trackPointer
              showBadge={false}
              title={expert.name}
            />
          </span>
        )}

        <span
          aria-hidden
          data-uploading={isUploading || undefined}
          className="pointer-events-none absolute inset-0 flex items-center justify-center rounded-full bg-foreground/45 text-background opacity-0 backdrop-blur-[2px] transition-opacity duration-150 group-hover:opacity-100 group-focus-visible:opacity-100 data-[uploading]:opacity-100"
        >
          {isUploading ? (
            <Icon icon={Loading03Icon} size={20} className="animate-spin" />
          ) : (
            <Icon icon={Camera01Icon} size={20} />
          )}
        </span>
      </button>
      <input
        ref={fileRef}
        type="file"
        aria-label={`Upload ${expert.name} photo`}
        accept="image/png,image/jpeg,image/webp,image/gif"
        className="hidden"
        onChange={handleChange}
      />
    </>
  );
}
