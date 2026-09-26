"use client";

import { motion, useReducedMotion } from "framer-motion";

import { Text } from "@/components/atoms/Text/Text";
import { NotificationSettingsControls } from "@/components/layout/NotificationSettings/NotificationSettingsControls";

import { EASE_OUT } from "../../helpers";

interface Props {
  index?: number;
}

export function BrowserNotificationsCard({ index = 0 }: Props) {
  const reduceMotion = useReducedMotion();

  return (
    <motion.section
      initial={reduceMotion ? false : { opacity: 0, y: 12 }}
      animate={reduceMotion ? undefined : { opacity: 1, y: 0 }}
      transition={
        reduceMotion
          ? undefined
          : { duration: 0.32, ease: EASE_OUT, delay: 0.04 + index * 0.05 }
      }
      className="flex w-full flex-col"
    >
      <div className="flex h-fit flex-col gap-4 rounded-[18px] border border-zinc-200 bg-white px-4 py-3 shadow-[0_1px_2px_rgba(15,15,20,0.04)]">
        <div className="flex flex-col gap-1">
          <Text variant="body-medium" as="span" className="text-textBlack">
            Browser
          </Text>
          {/* Unlike the email settings above, these live in this browser's
              local storage — they don't follow the account to another device. */}
          <Text variant="small" as="span" className="text-zinc-500">
            Set per device. Turning these on here won&apos;t carry over to your
            phone or another browser.
          </Text>
        </div>

        <NotificationSettingsControls />
      </div>
    </motion.section>
  );
}
