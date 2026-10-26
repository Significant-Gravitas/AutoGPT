"use client";

import Link, { useLinkStatus } from "next/link";
import { motion, useReducedMotion, type Variants } from "framer-motion";
import { cn } from "@/lib/utils";
import { Text } from "@/components/atoms/Text/Text";
import { LoadingSpinner } from "@/components/atoms/LoadingSpinner/LoadingSpinner";
import { Icon as UIIcon } from "@/components/atoms/Icon/Icon";
import type { IconSvgElement } from "@hugeicons/react";

export interface SectionNavLink {
  label: string;
  href: string;
  Icon: IconSvgElement;
}

type Props = {
  item: SectionNavLink;
  isActive: boolean;
};

function NavItemContent({
  label,
  Icon,
  isActive,
}: {
  label: string;
  Icon: IconSvgElement;
  isActive: boolean;
}) {
  const { pending } = useLinkStatus();

  return (
    <>
      <UIIcon
        icon={Icon}
        size={16}
        // TODO(tokens): no exact role for text-[#1F1F20] (#1F1F20)
        className={isActive ? "text-foreground-base" : "text-[#1F1F20]"}
      />
      <Text
        variant="body"
        as="span"
        className={cn(
          "flex-1",
          isActive
            ? // TODO(tokens): no exact role for text-[#1F1F20] (#1F1F20)
              "font-medium text-[#1F1F20]"
            : "font-normal text-foreground-body",
        )}
      >
        {label}
      </Text>
      {pending ? <LoadingSpinner size="small" /> : null}
    </>
  );
}

export function SectionNavItem({ item, isActive }: Props) {
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
          "flex h-[38px] w-full items-center gap-2 rounded-[8px] px-3 text-foreground-body transition-colors",
          // TODO(tokens): no exact role for hover:bg-[#F5F5F6] (#F5F5F6)
          isActive ? "bg-surface-hover" : "hover:bg-[#F5F5F6]",
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
