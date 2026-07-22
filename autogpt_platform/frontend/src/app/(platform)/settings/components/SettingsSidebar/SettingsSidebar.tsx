"use client";

import { motion, useReducedMotion } from "framer-motion";
import Link from "next/link";
import { ArrowLeftIcon } from "@phosphor-icons/react";
import { Text } from "@/components/atoms/Text/Text";
import { useSettingsSidebar } from "./useSettingsSidebar";
import { SettingsNavItem } from "./SettingsNavItem";

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
      className="hidden h-full w-[237px] shrink-0 overflow-y-auto border-r border-[#DADADC] bg-[#F9F9FA] px-[10px] pt-[13px] md:block"
    >
      <Link
        href="/copilot"
        aria-label="Back to home"
        className="mb-[16px] flex w-fit items-center gap-2 rounded-[8px] px-4 py-1 text-[#505057] transition-colors hover:text-[#1F1F20]"
      >
        <ArrowLeftIcon size={16} weight="bold" />
        <Text variant="body" as="span" className="font-medium">
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
