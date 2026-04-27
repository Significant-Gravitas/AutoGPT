"use client";

import { useRef, useState } from "react";
import { motion, useReducedMotion } from "framer-motion";
import { CameraIcon, UserIcon } from "@phosphor-icons/react";

import {
  Avatar,
  AvatarFallback,
  AvatarImage,
} from "@/components/atoms/Avatar/Avatar";
import { Text } from "@/components/atoms/Text/Text";
import { CircleNotchIcon } from "@phosphor-icons/react/dist/ssr";

interface Props {
  avatarUrl: string;
  name: string;
  email?: string;
  isUploading: boolean;
  onUpload: (file: File) => Promise<string | null>;
}

const EASE_OUT = [0.16, 1, 0.3, 1] as const;

export function ProfileHeader({
  avatarUrl,
  name,
  email,
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
    if (file) {
      await onUpload(file);
    }
    if (fileRef.current) fileRef.current.value = "";
  }

  return (
    <motion.div
      initial={reduceMotion ? { opacity: 0 } : { opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.32, ease: EASE_OUT }}
      className="flex flex-col items-center gap-5 sm:flex-row sm:items-center sm:gap-6"
    >
      <button
        type="button"
        onClick={openFilePicker}
        onMouseEnter={() => setIsHovered(true)}
        onMouseLeave={() => setIsHovered(false)}
        disabled={isUploading}
        aria-label="Change profile photo"
        className="group relative h-[112px] w-[112px] shrink-0 cursor-pointer rounded-full outline-none transition-transform duration-150 ease-out focus-visible:ring-2 focus-visible:ring-purple-400 focus-visible:ring-offset-2 active:scale-[0.97] disabled:cursor-wait"
      >
        <Avatar className="h-[112px] w-[112px] ring-2 ring-white shadow-[0_8px_28px_-12px_rgba(15,15,20,0.18)]">
          <AvatarImage src={avatarUrl} alt={name || "Profile"} />
          <AvatarFallback className="bg-gradient-to-br from-zinc-100 to-zinc-200 text-zinc-500">
            <UserIcon size={48} weight="regular" />
          </AvatarFallback>
        </Avatar>

        <motion.div
          initial={false}
          animate={{
            opacity: isUploading || isHovered ? 1 : 0,
          }}
          transition={{ duration: 0.18, ease: EASE_OUT }}
          className="pointer-events-none absolute inset-0 flex items-center justify-center rounded-full bg-black/45 text-white backdrop-blur-[2px]"
        >
          {isUploading ? (
            <CircleNotchIcon
              size={26}
              weight="bold"
              className="animate-spin"
            />
          ) : (
            <CameraIcon size={26} weight="regular" />
          )}
        </motion.div>

        <input
          ref={fileRef}
          type="file"
          accept="image/png,image/jpeg,image/webp,image/gif"
          className="hidden"
          onChange={handleChange}
        />
      </button>

      <div className="flex min-w-0 flex-col items-center text-center sm:items-start sm:text-left">
        <Text variant="h4" as="h1" className="text-[#1F1F20]">
          {name?.trim() || "Your profile"}
        </Text>
        {email ? (
          <Text variant="body" className="mt-1 text-zinc-500">
            {email}
          </Text>
        ) : null}
        <Text variant="small" className="mt-2 max-w-[420px] text-zinc-500">
          This information appears on your public marketplace profile.
        </Text>
      </div>
    </motion.div>
  );
}
