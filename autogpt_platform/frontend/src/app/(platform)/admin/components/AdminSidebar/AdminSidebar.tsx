"use client";

import { SettingsNavItem } from "@/app/(platform)/settings/components/SettingsSidebar/SettingsNavItem";
import { Text } from "@/components/atoms/Text/Text";
import { Icon } from "@/components/atoms/Icon/Icon";
import { ArrowLeft02Icon } from "@hugeicons/core-free-icons";
import { motion, useReducedMotion } from "framer-motion";
import Link from "next/link";
import { useAdminSidebar } from "./useAdminSidebar";

export function AdminSidebar() {
  const { items } = useAdminSidebar();
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
      className="hidden h-full w-[268px] shrink-0 overflow-y-auto border-r border-zinc-200 bg-zinc-50 px-2.5 pt-[13px] md:block"
    >
      <Link
        href="/copilot"
        aria-label="Back to home"
        className="mb-4 flex w-fit items-center gap-2 rounded-lg px-4 py-1 text-zinc-700 transition-colors hover:text-black"
      >
        <Icon icon={ArrowLeft02Icon} size={16} />
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
