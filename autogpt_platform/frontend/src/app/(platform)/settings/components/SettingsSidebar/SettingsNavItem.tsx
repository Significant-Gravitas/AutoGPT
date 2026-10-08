"use client";

import Link, { useLinkStatus } from "next/link";
import { motion, useReducedMotion, type Variants } from "motion/react";
import { cn } from "@/lib/utils";
import { Text } from "@/components/atoms/Text/Text";
import { LoadingSpinner } from "@/components/atoms/LoadingSpinner/LoadingSpinner";
import { Icon as UIIcon } from "@/components/atoms/Icon/Icon";
import type { SettingsNavItem as SettingsNavItemType } from "./helpers";

interface Props {
  item: SettingsNavItemType;
  isActive: boolean;
}

function NavItemContent({
  label,
  Icon,
  isActive,
}: {
  label: string;
  Icon: SettingsNavItemType["Icon"];
  isActive: boolean;
}) {
  const { pending } = useLinkStatus();

  return (
    <>
      <UIIcon icon={Icon} size={16} className="text-black" />
      <Text
        variant={isActive ? "body-medium" : "body"}
        as="span"
        className={cn("flex-1", !isActive && "text-zinc-700")}
      >
        {label}
      </Text>
      {pending ? <LoadingSpinner size="small" /> : null}
    </>
  );
}

export function SettingsNavItem({ item, isActive }: Props) {
  const reduceMotion = useReducedMotion();

  const variants: Variants = reduceMotion
    ? {
        hidden: { opacity: 0 },
        show: { opacity: 1, transition: { duration: 0.15 } },
      }
    : {
        hidden: { opacity: 0, x: -6 },
        show: {
          opacity: 1,
          x: 0,
          transition: {
            duration: 0.22,
            ease: [0, 0, 0.2, 1] as const,
          },
        },
      };

  return (
    <motion.div variants={variants} className="w-full">
      <Link
        href={item.href}
        aria-current={isActive ? "page" : undefined}
        className={cn(
          "flex h-[38px] w-full items-center gap-2 rounded-lg px-3 text-zinc-700 transition-colors",
          isActive ? "bg-zinc-100" : "hover:bg-zinc-50",
        )}
      >
        <NavItemContent
          label={item.label}
          Icon={item.Icon}
          isActive={isActive}
        />
      </Link>
    </motion.div>
  );
}
