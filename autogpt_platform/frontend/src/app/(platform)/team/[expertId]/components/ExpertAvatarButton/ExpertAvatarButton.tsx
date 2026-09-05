"use client";

import { Expert } from "@/app/api/__generated__/models/expert";
import {
  Avatar,
  AvatarFallback,
  AvatarImage,
} from "@/components/atoms/Avatar/Avatar";
import { Icon } from "@/components/atoms/Icon/Icon";
import {
  Camera01Icon,
  Loading03Icon,
  PencilIcon,
} from "@hugeicons/core-free-icons";
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
        className="group relative size-24 shrink-0 cursor-pointer rounded-full outline-none transition-transform duration-150 ease-out focus-visible:ring-2 focus-visible:ring-zinc-400 focus-visible:ring-offset-2 active:scale-[0.97] disabled:cursor-wait"
      >
        <Avatar className="size-24 bg-white ring-4 ring-white">
          {expert.avatar_url ? (
            <AvatarImage
              src={expert.avatar_url}
              alt={expert.name}
              width={192}
              height={192}
            />
          ) : null}
          <AvatarFallback>{expert.name}</AvatarFallback>
        </Avatar>

        <span
          aria-hidden
          data-uploading={isUploading || undefined}
          className="pointer-events-none absolute inset-0 flex items-center justify-center rounded-full bg-black/45 text-white opacity-0 backdrop-blur-[2px] transition-opacity duration-150 group-hover:opacity-100 group-focus-visible:opacity-100 data-[uploading]:opacity-100"
        >
          {isUploading ? (
            <Icon icon={Loading03Icon} size={20} className="animate-spin" />
          ) : (
            <Icon icon={Camera01Icon} size={20} />
          )}
        </span>

        <span
          aria-hidden
          className="pointer-events-none absolute -bottom-0.5 -right-0.5 flex size-6 items-center justify-center rounded-full border-2 border-white bg-white text-black shadow-[0_3px_10px_-2px_rgba(15,15,20,0.25)]"
        >
          <Icon icon={PencilIcon} size={12} />
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
