"use client";

import { motion, useReducedMotion } from "motion/react";
import Link from "next/link";
import { Text } from "@/components/atoms/Text/Text";
import { useSettingsSidebar } from "./useSettingsSidebar";
import { SettingsNavItem } from "./SettingsNavItem";
import { ArrowLeft02Icon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";

export function SettingsSidebar() {
  const { items } = useSettingsSidebar();
  const reduceMotion = useReducedMotion();

  const container = {
    hidden: {},
    show: {
      transition: {
        staggerChildren: reduceMotion ? 0 : 0.04,
        delayChildren: 0.08,
      },
    },
  };

  return (
    <motion.aside
      initial={reduceMotion ? { opacity: 0 } : { opacity: 0, x: -10 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.25, ease: [0, 0, 0.2, 1] as const }}
      className="hidden h-full w-[237px] shrink-0 overflow-y-auto border-r border-zinc-200 bg-zinc-50 px-2.5 pt-[13px] md:block"
    >
      <Link
        href="/copilot"
        aria-label="Back to home"
        className="mb-4 flex w-fit items-center gap-2 rounded-lg px-4 py-1 text-zinc-700 transition-colors hover:text-black"
      >
        <Icon icon={ArrowLeft02Icon} size={16} />
        <Text variant="body-medium" as="span">
          Back
        </Text>
      </Link>
      <motion.nav
        variants={container}
        initial="hidden"
        animate="show"
        className="flex flex-col items-start gap-[7px]"
      >
        {items.map((item) => (
          <SettingsNavItem
            key={item.href}
            item={item}
            isActive={item.isActive}
          />
        ))}
      </motion.nav>
    </motion.aside>
  );
}
