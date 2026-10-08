import { Button } from "@/components/atoms/Button/Button";
import { cn } from "@/lib/utils";
import { AnimatePresence, motion } from "motion/react";

import React, { ButtonHTMLAttributes, useState } from "react";
import { Cancel01Icon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";

interface Props extends ButtonHTMLAttributes<HTMLButtonElement> {
  selected?: boolean;
  number?: number;
  name?: string;
}

export const FilterChip: React.FC<Props> = ({
  selected = false,
  number,
  name,
  className,
  ...rest
}) => {
  const [isHovered, setIsHovered] = useState(false);
  return (
    <AnimatePresence mode="wait">
      <Button
        variant="ghost"
        size="md"
        onMouseEnter={() => setIsHovered(true)}
        onMouseLeave={() => setIsHovered(false)}
        className={cn(
          "group h-9 w-fit min-w-0 gap-1 rounded-3xl border border-zinc-300 bg-transparent px-2.5 py-1.5 shadow-none",
          "hover:border-purple-500 hover:bg-transparent focus:ring-0 disabled:cursor-not-allowed disabled:opacity-50",
          selected && "border-0 bg-purple-700 hover:border",
          className,
        )}
        {...rest}
      >
        <span
          className={cn(
            "font-sans text-sm leading-5.5 font-medium text-zinc-600 group-hover:text-zinc-600 group-disabled:text-zinc-400",
            selected && "text-zinc-50",
          )}
        >
          {name}
        </span>
        {selected && !isHovered && (
          <motion.span
            initial={{ opacity: 0.5, scale: 0.5, filter: "blur(20px)" }}
            animate={{ opacity: 1, scale: 1, filter: "blur(0px)" }}
            exit={{ opacity: 0.5, scale: 0.5, filter: "blur(20px)" }}
            transition={{ duration: 0.3, type: "spring", bounce: 0.2 }}
            className="flex h-4 w-4 items-center justify-center rounded-full bg-zinc-50"
          >
            <Icon icon={Cancel01Icon} size={12} className="text-purple-700" />
          </motion.span>
        )}
        {number !== undefined && isHovered && (
          <motion.span
            initial={{ opacity: 0.5, scale: 0.5, filter: "blur(10px)" }}
            animate={{ opacity: 1, scale: 1, filter: "blur(0px)" }}
            exit={{ opacity: 0.5, scale: 0.5, filter: "blur(10px)" }}
            transition={{ duration: 0.3, type: "spring", bounce: 0.2 }}
            className="flex h-5.5 items-center rounded-2xl bg-purple-700 p-1.5 text-zinc-50"
          >
            {number > 100 ? "100+" : number}
          </motion.span>
        )}
      </Button>
    </AnimatePresence>
  );
};
