import { Text } from "@/components/atoms/Text/Text";
import { blockMenuButtonBase } from "./style";
import { cn } from "@/lib/utils";
import { AnimatePresence, motion } from "framer-motion";

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
      <button
        type="button"
        onMouseEnter={() => setIsHovered(true)}
        onMouseLeave={() => setIsHovered(false)}
        className={cn(
          blockMenuButtonBase,
          "group h-9 w-fit space-x-1 rounded-3xl border border-border-strong bg-transparent px-2.5 py-1.5",
          "hover:border-brand hover:bg-transparent disabled:cursor-not-allowed",
          selected && "border-0 bg-accent-strong hover:border",
          className,
        )}
        {...rest}
      >
        <Text
          variant="body-medium"
          as="span"
          tone="secondary"
          className={cn(
            "group-hover:text-foreground-secondary group-disabled:text-foreground-subtle",
            selected && "text-action-foreground",
          )}
        >
          {name}
        </Text>
        {selected && !isHovered && (
          <motion.span
            initial={{ opacity: 0.5, scale: 0.5, filter: "blur(20px)" }}
            animate={{ opacity: 1, scale: 1, filter: "blur(0px)" }}
            exit={{ opacity: 0.5, scale: 0.5, filter: "blur(20px)" }}
            transition={{ duration: 0.3, type: "spring", bounce: 0.2 }}
            className="flex h-4 w-4 items-center justify-center rounded-full bg-surface-muted"
          >
            <Icon
              icon={Cancel01Icon}
              size={12}
              className="text-accent-strong"
            />
          </motion.span>
        )}
        {number !== undefined && isHovered && (
          <motion.span
            initial={{ opacity: 0.5, scale: 0.5, filter: "blur(10px)" }}
            animate={{ opacity: 1, scale: 1, filter: "blur(0px)" }}
            exit={{ opacity: 0.5, scale: 0.5, filter: "blur(10px)" }}
            transition={{ duration: 0.3, type: "spring", bounce: 0.2 }}
            className="flex h-[1.375rem] items-center rounded-full bg-accent-strong p-1.5 text-action-foreground"
          >
            {number > 100 ? "100+" : number}
          </motion.span>
        )}
      </button>
    </AnimatePresence>
  );
};
