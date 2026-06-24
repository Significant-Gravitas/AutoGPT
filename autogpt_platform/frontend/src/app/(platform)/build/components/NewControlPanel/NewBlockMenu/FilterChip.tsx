import { Button } from "@/components/__legacy__/ui/button";
import { cn } from "@/lib/utils";
import { XIcon } from "@phosphor-icons/react";
import { AnimatePresence, motion } from "framer-motion";

import React, { ButtonHTMLAttributes, useState } from "react";

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
        onMouseEnter={() => setIsHovered(true)}
        onMouseLeave={() => setIsHovered(false)}
        className={cn(
          "group w-fit space-x-1 rounded-[1.5rem] border border-zinc-300 bg-transparent px-[0.625rem] py-[0.375rem] shadow-none",
          "hover:border-violet-500 hover:bg-transparent focus:ring-0 disabled:cursor-not-allowed",
          selected && "border-0 bg-violet-700 hover:border",
          className,
        )}
        {...rest}
      >
        <span
          className={cn(
            "font-sans text-sm font-medium leading-[1.375rem] text-zinc-600 group-hover:text-zinc-600 group-disabled:text-zinc-400",
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
            <XIcon size={12} weight="bold" className="text-violet-700" />
          </motion.span>
        )}
        {number !== undefined && isHovered && (
          <motion.span
            initial={{ opacity: 0.5, scale: 0.5, filter: "blur(10px)" }}
            animate={{ opacity: 1, scale: 1, filter: "blur(0px)" }}
            exit={{ opacity: 0.5, scale: 0.5, filter: "blur(10px)" }}
            transition={{ duration: 0.3, type: "spring", bounce: 0.2 }}
            className="flex h-[1.375rem] items-center rounded-[1.25rem] bg-violet-700 p-[0.375rem] text-zinc-50"
          >
            {number > 100 ? "100+" : number}
          </motion.span>
        )}
      </Button>
    </AnimatePresence>
  );
};
