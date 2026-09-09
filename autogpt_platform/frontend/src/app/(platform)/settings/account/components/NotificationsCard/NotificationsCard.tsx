"use client";

import { motion, useReducedMotion } from "framer-motion";

import { Select } from "@/components/atoms/Select/Select";
import { Switch } from "@/components/atoms/Switch/Switch";
import { Text } from "@/components/atoms/Text/Text";

import {
  BRIEFING_OPTIONS,
  EASE_OUT,
  type BriefingFrequency,
  type NotificationSettings,
} from "../../helpers";

interface Props {
  values: NotificationSettings;
  onBriefingFrequencyChange: (value: BriefingFrequency) => void;
  onAlertsChange: (value: boolean) => void;
  onStoreVerdictsChange: (value: boolean) => void;
  index?: number;
}

export function NotificationsCard({
  values,
  onBriefingFrequencyChange,
  onAlertsChange,
  onStoreVerdictsChange,
  index = 0,
}: Props) {
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
            Email
          </Text>
          <Text variant="small" as="span" className="text-zinc-500">
            Billing and account messages are always sent — they are about your
            account, not a promotion.
          </Text>
        </div>

        <div className="flex flex-col gap-2">
          <Select
            id="briefing-frequency"
            label="Briefing"
            value={values.briefingFrequency}
            options={BRIEFING_OPTIONS.map((option) => ({
              value: option.value,
              label: option.label,
            }))}
            onValueChange={(value) =>
              onBriefingFrequencyChange(value as BriefingFrequency)
            }
          />
          <Text variant="small" as="span" className="text-zinc-500">
            What your agents got done, at around 07:30 your time. Never sent
            when nothing ran.
          </Text>
        </div>

        <div className="flex items-center justify-between gap-4">
          <div className="flex flex-col">
            <Text variant="body-medium" as="span" className="text-textBlack">
              Alerts
            </Text>
            <Text variant="small" as="span" className="text-zinc-500">
              Only when something is waiting on you — never for a successful
              run. At most two a day.
            </Text>
          </div>
          <Switch
            checked={values.alertsEnabled}
            onCheckedChange={onAlertsChange}
            aria-label="Alerts"
          />
        </div>

        <div className="flex items-center justify-between gap-4">
          <div className="flex flex-col">
            <Text variant="body-medium" as="span" className="text-textBlack">
              Marketplace reviews
            </Text>
            <Text variant="small" as="span" className="text-zinc-500">
              When an agent you submitted is approved or needs changes.
            </Text>
          </div>
          <Switch
            checked={values.storeVerdictsEnabled}
            onCheckedChange={onStoreVerdictsChange}
            aria-label="Marketplace reviews"
          />
        </div>
      </div>
    </motion.section>
  );
}
