"use client";

import { ClockIcon, GlobeIcon, InfoIcon } from "@phosphor-icons/react";
import { AnimatePresence, motion, useReducedMotion } from "framer-motion";

import { Select } from "@/components/atoms/Select/Select";
import { Text } from "@/components/atoms/Text/Text";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/atoms/Tooltip/BaseTooltip";

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
  const browserTz = detectBrowserTimezone();
  const offset = formatGmtOffset(value);
  const browserLabel = findTimezoneLabel(browserTz);

  const showAutoDetect = initialValue !== browserTz && value !== browserTz;

  const options = TIMEZONES.some((t) => t.value === value)
    ? TIMEZONES
    : [{ value, label: findTimezoneLabel(value) }, ...TIMEZONES];

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
      <div className="flex h-fit flex-col justify-center gap-3 rounded-[18px] border border-zinc-200 bg-white px-4 py-3 shadow-[0_1px_2px_rgba(15,15,20,0.04)]">
        <div className="flex items-center justify-between gap-4">
          <div className="flex items-center gap-2">
            <Text variant="body-medium" as="span" className="text-textBlack">
              Time zone
            </Text>
            <Tooltip>
              <TooltipTrigger asChild>
                <button
                  type="button"
                  aria-label="Time zone info"
                  className="flex items-center text-zinc-400 transition-colors hover:text-zinc-600 focus-visible:text-zinc-600 focus-visible:outline-none"
                >
                  <InfoIcon size={16} weight="duotone" />
                </button>
              </TooltipTrigger>
              <TooltipContent>
                Used for run schedules, summaries, and timestamps.
              </TooltipContent>
            </Tooltip>
          </div>

          <div className="flex h-fit items-center gap-2">
            <Select
              id="timezone"
              label="Timezone"
              hideLabel
              value={value}
              onValueChange={onChange}
              options={options}
              placeholder="Select your timezone"
              size="small"
              wrapperClassName="!mb-0 w-fit"
            />
            {offset ? (
              <div className="flex h-fit shrink-0 items-center gap-1.5 rounded-full border border-zinc-200 bg-zinc-50 px-3 py-1.5 tabular-nums text-zinc-600">
                <ClockIcon size={14} weight="duotone" />
                <Text
                  variant="small-medium"
                  as="span"
                  className="text-zinc-600"
                >
                  {offset}
                </Text>
              </div>
            ) : null}
          </div>
        </div>

        <AnimatePresence initial={false}>
          {showAutoDetect ? (
            <motion.button
              key="autodetect"
              type="button"
              onClick={() => onChange(browserTz)}
              initial={reduceMotion ? false : { opacity: 0, y: -4, height: 0 }}
              animate={
                reduceMotion ? undefined : { opacity: 1, y: 0, height: "auto" }
              }
              exit={reduceMotion ? undefined : { opacity: 0, y: -4, height: 0 }}
              transition={
                reduceMotion ? undefined : { duration: 0.22, ease: EASE_OUT }
              }
              className="flex w-fit items-center gap-2 self-end overflow-hidden rounded-full border border-violet-300 bg-violet-50/60 px-3 py-1.5 text-left text-violet-800 transition-colors duration-150 ease-out hover:bg-violet-50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-violet-400 focus-visible:ring-offset-2"
            >
              <GlobeIcon size={16} weight="duotone" />
              <Text variant="small" as="span" className="text-violet-800">
                Looks like you&apos;re in{" "}
                <span className="font-medium">{browserLabel}</span>. Use that?
              </Text>
            </motion.button>
          ) : null}
        </AnimatePresence>
      </div>
    </motion.section>
  );
}
