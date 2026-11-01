"use client";

import { useRef, useState } from "react";
import { motion, useReducedMotion } from "framer-motion";
import {
  Avatar,
  AvatarFallback,
  AvatarImage,
} from "@/components/atoms/Avatar/Avatar";
import { Input } from "@/components/atoms/Input/Input";
import { Text } from "@/components/atoms/Text/Text";

import { type ProfileFormState, validateForm } from "../../helpers";
import {
  Camera01Icon,
  Loading03Icon,
  PencilIcon,
  UserIcon,
} from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";

interface Props {
  avatarUrl: string;
  name: string;
  username: string;
  errors: ReturnType<typeof validateForm>["errors"];
  onChange: <K extends keyof ProfileFormState>(
    key: K,
    value: ProfileFormState[K],
  ) => void;
  isUploading: boolean;
  onUpload: (file: File) => Promise<string | null>;
}

const EASE_OUT = [0.16, 1, 0.3, 1] as const;
// iOS-style cubic-bezier — gives the badge a smooth in-and-out feel as it
// sweeps along the avatar's circumference (Emil's drawer easing).
const BADGE_ARC_EASE = [0.32, 0.72, 0, 1] as const;

export function ProfileHeader({
  avatarUrl,
  name,
  username,
  errors,
  onChange,
  isUploading,
  onUpload,
}: Props) {
  const fileRef = useRef<HTMLInputElement>(null);
  const [isHovered, setIsHovered] = useState(false);
  const reduceMotion = useReducedMotion();

  function openFilePicker() {
    if (isUploading) return;
    fileRef.current?.click();
  }

  async function handleChange(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    e.target.value = "";
    if (file) {
      await onUpload(file);
    }
  }

  return (
    <motion.div
      initial={reduceMotion ? { opacity: 0 } : { opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.32, ease: EASE_OUT }}
      className="flex flex-col items-center gap-5 sm:flex-row sm:items-start sm:gap-6"
    >
      <button
        type="button"
        onClick={openFilePicker}
        onMouseEnter={() => setIsHovered(true)}
        onMouseLeave={() => setIsHovered(false)}
        disabled={isUploading}
        aria-label="Change profile photo"
        className="group relative h-[112px] w-[112px] shrink-0 cursor-pointer rounded-full transition-transform duration-150 ease-out outline-none focus-visible:ring-2 focus-visible:ring-brand/80 focus-visible:ring-offset-2 active:scale-[0.97] disabled:cursor-wait"
      >
        {/* oxlint-disable-next-line design-system/no-restyle, design-system/no-arbitrary-values -- the profile photo floats on a soft shadow with a white rim */}
        <Avatar className="h-[112px] w-[112px] shadow-[0_8px_28px_-12px_rgba(15,15,20,0.18)] ring-2 ring-surface">
          <AvatarImage
            src={avatarUrl}
            alt={name || "Profile"}
            width={224}
            height={224}
          />
          {/* oxlint-disable-next-line design-system/no-restyle -- the empty photo is a grey gradient behind a person glyph */}
          <AvatarFallback className="bg-gradient-to-br from-surface-hover to-surface-active text-foreground-tertiary">
            <Icon icon={UserIcon} size={48} />
          </AvatarFallback>
        </Avatar>

        <motion.div
          initial={false}
          animate={{
            opacity: isUploading || isHovered ? 1 : 0,
          }}
          transition={{ duration: 0.18, ease: EASE_OUT }}
          // oxlint-disable-next-line design-system/no-arbitrary-values -- a 2px blur under the hover veil; backdrop-blur-sm is 4px
          className="pointer-events-none absolute inset-0 flex items-center justify-center rounded-full bg-black/45 text-action-foreground backdrop-blur-[2px]"
        >
          {isUploading ? (
            // oxlint-disable-next-line design-system/no-restyle -- the upload glyph spins while the photo uploads
            <Icon icon={Loading03Icon} size={26} className="animate-spin" />
          ) : (
            <Icon icon={Camera01Icon} size={26} />
          )}
        </motion.div>

        <motion.span
          aria-hidden
          initial={false}
          animate={{ rotate: !reduceMotion && isHovered ? -90 : 0 }}
          transition={{
            duration: reduceMotion ? 0 : 0.32,
            ease: BADGE_ARC_EASE,
          }}
          className="pointer-events-none absolute inset-0 origin-center"
        >
          <motion.span
            initial={false}
            animate={{ rotate: !reduceMotion && isHovered ? 90 : 0 }}
            transition={{
              duration: reduceMotion ? 0 : 0.32,
              ease: BADGE_ARC_EASE,
            }}
            // oxlint-disable-next-line design-system/no-arbitrary-values -- the edit badge's lift shadow has no token
            className="absolute right-1 bottom-1 flex h-9 w-9 origin-center items-center justify-center rounded-full border-2 border-surface bg-surface text-foreground-base shadow-[0_3px_10px_-2px_rgba(15,15,20,0.25)]"
          >
            <Icon icon={PencilIcon} size={16} />
          </motion.span>
        </motion.span>

        <input
          ref={fileRef}
          type="file"
          accept="image/png,image/jpeg,image/webp,image/gif"
          className="hidden"
          onChange={handleChange}
        />
      </button>

      <div className="grid w-full gap-4 sm:grid-cols-2">
        <label htmlFor="profile-name" className="flex w-full flex-col gap-2">
          <Text variant="body-medium" as="span" className="mx-4">
            Display name
          </Text>
          <Input
            id="profile-name"
            label="Display name"
            hideLabel
            placeholder="Jane Doe"
            value={name}
            error={errors.name}
            onChange={(e) => onChange("name", e.target.value)}
          />
        </label>

        <label
          htmlFor="profile-username"
          className="flex w-full flex-col gap-2"
        >
          <div className="flex items-center justify-between px-4">
            <Text variant="body-medium" as="span">
              Handle
            </Text>
            <Text variant="small" as="span" tone="subtle">
              agpt.co/@your-handle
            </Text>
          </div>
          <Input
            id="profile-username"
            label="Handle"
            hideLabel
            placeholder="jane_doe"
            value={username}
            error={errors.username}
            onChange={(e) => onChange("username", e.target.value)}
          />
        </label>
      </div>
    </motion.div>
  );
}
