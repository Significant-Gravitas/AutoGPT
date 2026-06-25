"use client";

import { motion, useReducedMotion } from "framer-motion";
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
      <Text
        variant="label"
        as="span"
        className="mb-[16px] block px-4 font-medium text-[#27272a]"
      >
        SETTINGS
      </Text>
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
