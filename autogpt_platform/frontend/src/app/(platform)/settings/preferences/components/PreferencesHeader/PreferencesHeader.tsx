"use client";

import { motion, useReducedMotion } from "framer-motion";

import { Text } from "@/components/atoms/Text/Text";

import { EASE_OUT } from "../../helpers";

export function PreferencesHeader() {
  const reduceMotion = useReducedMotion();

  return (
    <motion.header
      initial={reduceMotion ? false : { opacity: 0, y: 8 }}
      animate={reduceMotion ? undefined : { opacity: 1, y: 0 }}
      transition={reduceMotion ? undefined : { duration: 0.32, ease: EASE_OUT }}
      className="flex min-w-0 flex-col pb-2 pl-4"
    >
      <Text variant="h4" as="h1" className="leading-[28px] text-textBlack">
        Preferences
      </Text>
      <Text variant="body" className="mt-4 max-w-[600px] text-zinc-700">
        Tune your account, time zone, and which notifications reach you.
      </Text>
    </motion.header>
  );
}
