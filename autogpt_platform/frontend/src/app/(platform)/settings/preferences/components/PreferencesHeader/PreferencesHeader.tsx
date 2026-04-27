"use client";

import { motion, useReducedMotion } from "framer-motion";

import { Text } from "@/components/atoms/Text/Text";

import { EASE_OUT } from "../../helpers";

export function PreferencesHeader() {
  const reduceMotion = useReducedMotion();

  return (
    <motion.header
      initial={reduceMotion ? { opacity: 0 } : { opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.32, ease: EASE_OUT }}
      className="flex flex-col gap-2"
    >
      <Text
        variant="h3"
        as="h1"
        className="bg-gradient-to-br from-[#1F1F20] via-[#3a3a3f] to-[#7a4dff] bg-clip-text text-transparent"
      >
        Settings
      </Text>
      <Text variant="large" className="text-zinc-500">
        Tune your account, time zone, and which notifications reach you.
      </Text>
    </motion.header>
  );
}
