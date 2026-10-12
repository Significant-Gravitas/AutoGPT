"use client";

import { motion, useReducedMotion } from "framer-motion";

import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { openConsentSettings } from "@/services/consent/consent";
import { useConsentManagerStatus } from "@/services/consent/useConsent";

import { EASE_OUT } from "../../helpers";

interface Props {
  index?: number;
}

export function CookieSettingsCard({ index = 0 }: Props) {
  const reduceMotion = useReducedMotion();
  const status = useConsentManagerStatus();

  return (
    <motion.section
      initial={reduceMotion ? false : { opacity: 0, y: 12 }}
      animate={reduceMotion ? undefined : { opacity: 1, y: 0 }}
      transition={
        reduceMotion
          ? undefined
          : {
              duration: 0.32,
              ease: EASE_OUT,
              delay: 0.04 + index * 0.05,
            }
      }
      className="flex w-full flex-col"
    >
      <div className="flex items-center justify-between gap-4 rounded-[18px] border border-zinc-200 bg-white px-4 py-4 shadow-[0_1px_2px_rgba(15,15,20,0.04)]">
        <div className="flex min-w-0 flex-col gap-0.5">
          <Text variant="body-medium" as="span" className="text-textBlack">
            Cookies
          </Text>
          <Text variant="small" as="span" className="text-zinc-500">
            Choose which optional cookies AutoGPT may use.
          </Text>
        </div>

        {status === "unavailable" ? (
          <Text
            variant="small"
            as="span"
            role="status"
            className="max-w-64 text-right text-zinc-500"
          >
            Cookie settings couldn&apos;t load. A content blocker may be
            stopping them.
          </Text>
        ) : (
          <Button
            type="button"
            size="small"
            variant="secondary"
            disabled={status !== "ready"}
            onClick={openConsentSettings}
          >
            Cookie settings
          </Button>
        )}
      </div>
    </motion.section>
  );
}
