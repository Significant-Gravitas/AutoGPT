"use client";

import { Icon } from "@/components/atoms/Icon/Icon";
import { ExpertAvatar } from "@/components/molecules/ExpertAvatar/ExpertAvatar";
import { Camera01Icon, Cancel01Icon } from "@hugeicons/core-free-icons";
import Image from "next/image";
import { useRef, type ChangeEvent } from "react";
import { ACCEPTED_IMAGE_TYPES } from "../helpers";

interface Props {
  name: string;
  imageUrl: string | null;
  onPick: (event: ChangeEvent<HTMLInputElement>) => void;
  onRemove: () => void;
}

/** The optional photo: the default avatar until a picture is picked, with a
 *  camera badge saying it can be changed. */
export function AvatarSlot({ name, imageUrl, onPick, onRemove }: Props) {
  const inputRef = useRef<HTMLInputElement>(null);

  return (
    <div className="relative shrink-0">
      <button
        type="button"
        onClick={() => inputRef.current?.click()}
        aria-label={imageUrl ? "Change photo" : "Add a photo (optional)"}
        className="group relative flex size-14 items-center justify-center rounded-full outline-none focus-visible:ring-2 focus-visible:ring-foreground-subtle focus-visible:ring-offset-2"
      >
        {imageUrl ? (
          <Image
            src={imageUrl}
            alt=""
            width={56}
            height={56}
            // A local blob: preview of the picked file; nothing to optimise.
            unoptimized
            className="size-14 rounded-full object-cover ring-1 ring-border-base"
          />
        ) : (
          <ExpertAvatar name={name || null} avatarUrl={null} size={56} />
        )}
        <span className="absolute -right-0.5 -bottom-0.5 flex size-6 items-center justify-center rounded-full bg-surface text-foreground-secondary shadow-sm ring-1 ring-border-base transition-colors group-hover:text-foreground-strong">
          <Icon icon={Camera01Icon} size={13} aria-hidden />
        </span>
      </button>
      {imageUrl ? (
        <button
          type="button"
          onClick={onRemove}
          aria-label="Remove photo"
          className="absolute -top-1 -left-1 flex size-5 items-center justify-center rounded-full bg-surface text-foreground-secondary shadow-sm ring-1 ring-border-base before:absolute before:-inset-2.5 hover:text-foreground-strong"
        >
          <Icon icon={Cancel01Icon} size={11} aria-hidden />
        </button>
      ) : null}
      <input
        ref={inputRef}
        type="file"
        accept={ACCEPTED_IMAGE_TYPES}
        className="sr-only"
        tabIndex={-1}
        aria-hidden
        data-testid="raise-expert-photo-input"
        onChange={onPick}
      />
    </div>
  );
}
