"use client";

import { useMemo } from "react";
import { ClockIcon, GlobeIcon } from "@phosphor-icons/react";
import { AnimatePresence, motion, useReducedMotion } from "framer-motion";

import { Select } from "@/components/atoms/Select/Select";
import { Text } from "@/components/atoms/Text/Text";

import {
  EASE_OUT,
  TIMEZONES,
  detectBrowserTimezone,
  findTimezoneLabel,
  formatGmtOffset,
} from "../../helpers";

interface Props {
  value: string;
  initialValue: string;
  onChange: (timezone: string) => void;
  index?: number;
}

export function TimezoneCard({
  value,
  initialValue,
  onChange,
  index = 0,
}: Props) {
  const reduceMotion = useReducedMotion();
  const browserTz = useMemo(detectBrowserTimezone, []);
  const offset = useMemo(() => formatGmtOffset(value), [value]);
  const browserLabel = useMemo(
    () => findTimezoneLabel(browserTz),
    [browserTz],
  );

  const showAutoDetect = initialValue !== browserTz && value !== browserTz;
  const isDirty = value !== initialValue;

  const options = useMemo(() => {
    if (TIMEZONES.some((t) => t.value === value)) return TIMEZONES;
    return [{ value, label: findTimezoneLabel(value) }, ...TIMEZONES];
  }, [value]);

  return (
    <motion.section
      initial={reduceMotion ? { opacity: 0 } : { opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.32, ease: EASE_OUT, delay: 0.04 + index * 0.05 }}
      className="rounded-[18px] border border-zinc-200 bg-white p-6 shadow-[0_1px_2px_rgba(15,15,20,0.04)] transition-shadow duration-200 ease-out focus-within:shadow-[0_8px_28px_-12px_rgba(15,15,20,0.12)]"
    >
      <div className="flex items-start justify-between gap-4">
        <div className="flex flex-col gap-1">
          <Text
            variant="small-medium"
            as="span"
            className="uppercase tracking-[0.08em] text-zinc-400"
          >
            Time & Locale
          </Text>
          <Text variant="h4" as="h2" className="text-[#1F1F20]">
            Time zone
          </Text>
          <Text variant="small" className="text-zinc-500">
            We'll use this for run schedules, summaries, and timestamps you see.
          </Text>
        </div>

        <motion.div
          initial={false}
          animate={{
            opacity: offset ? 1 : 0,
            scale: offset ? 1 : 0.96,
          }}
          transition={{ duration: 0.18, ease: EASE_OUT }}
          aria-hidden={!offset}
          className="flex shrink-0 items-center gap-1.5 rounded-full border border-zinc-200 bg-zinc-50 px-3 py-1.5 tabular-nums text-zinc-600"
        >
          <ClockIcon size={14} weight="duotone" />
          <Text variant="small-medium" as="span" className="text-zinc-600">
            {offset ?? ""}
          </Text>
        </motion.div>
      </div>

      <div className="mt-5">
        <Select
          id="timezone"
          label="Timezone"
          hideLabel
          value={value}
          onValueChange={onChange}
          options={options}
          placeholder="Select your timezone"
          size="medium"
        />
      </div>

      <AnimatePresence initial={false}>
        {showAutoDetect ? (
          <motion.button
            key="autodetect"
            type="button"
            onClick={() => onChange(browserTz)}
            initial={
              reduceMotion ? { opacity: 0 } : { opacity: 0, y: -4, height: 0 }
            }
            animate={{ opacity: 1, y: 0, height: "auto" }}
            exit={
              reduceMotion ? { opacity: 0 } : { opacity: 0, y: -4, height: 0 }
            }
            transition={{ duration: 0.22, ease: EASE_OUT }}
            className="mt-1 flex w-full items-center gap-2 overflow-hidden rounded-xl border border-dashed border-violet-300 bg-violet-50/60 px-3 py-2 text-left text-violet-800 transition-colors duration-150 ease-out hover:bg-violet-50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-violet-400 focus-visible:ring-offset-2"
          >
            <GlobeIcon size={16} weight="duotone" />
            <Text variant="small" as="span" className="text-violet-800">
              Looks like you're in{" "}
              <span className="font-medium">{browserLabel}</span>. Use that?
            </Text>
          </motion.button>
        ) : null}
      </AnimatePresence>

      <AnimatePresence initial={false}>
        {isDirty ? (
          <motion.div
            key="changed"
            initial={reduceMotion ? { opacity: 0 } : { opacity: 0, y: -4 }}
            animate={{ opacity: 1, y: 0 }}
            exit={reduceMotion ? { opacity: 0 } : { opacity: 0, y: -4 }}
            transition={{ duration: 0.18, ease: EASE_OUT }}
            className="mt-3 flex items-center gap-2"
          >
            <span
              aria-hidden
              className="h-1.5 w-1.5 rounded-full bg-amber-500"
            />
            <Text variant="small" as="span" className="text-zinc-500">
              Unsaved change — review and hit save below.
            </Text>
          </motion.div>
        ) : null}
      </AnimatePresence>
    </motion.section>
  );
}
