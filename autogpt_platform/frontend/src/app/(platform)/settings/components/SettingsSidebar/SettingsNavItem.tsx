"use client";

import Link, { useLinkStatus } from "next/link";
import { motion, useReducedMotion, type Variants } from "framer-motion";
import { cn } from "@/lib/utils";
import { Text } from "@/components/atoms/Text/Text";
import { LoadingSpinner } from "@/components/atoms/LoadingSpinner/LoadingSpinner";
import type { SettingsNavItem as SettingsNavItemType } from "./helpers";

type Props = {
  item: SettingsNavItemType;
  isActive: boolean;
};

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
      <Icon
        size={16}
        weight={isActive ? "regular" : "light"}
        className={isActive ? "text-black" : "text-[#1F1F20]"}
      />
      <Text
        variant="body"
        as="span"
        className={cn(
          "flex-1",
          isActive
            ? "font-medium text-[#1F1F20]"
            : "font-normal text-[#505057]",
        )}
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
    <motion.div variants={variants} className="w-[217px]">
      <Link
        href={item.href}
        aria-current={isActive ? "page" : undefined}
        className={cn(
          "flex h-[38px] w-[217px] items-center gap-2 rounded-[8px] px-3 text-[#505057] transition-colors",
          isActive ? "bg-[#EFEFF0]" : "hover:bg-[#F5F5F6]",
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
