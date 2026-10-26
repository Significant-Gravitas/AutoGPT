import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
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
      <Button
        variant="legacy-default"
        onMouseEnter={() => setIsHovered(true)}
        onMouseLeave={() => setIsHovered(false)}
        className={cn(
          "group w-fit space-x-1 rounded-[1.5rem] border border-border-strong bg-transparent px-[0.625rem] py-[0.375rem] shadow-none",
          // TODO(tokens): no exact role: hover:border-violet-500 #8B5CF6
          "hover:border-violet-500 hover:bg-transparent focus:ring-0 disabled:cursor-not-allowed",
          // TODO(tokens): no fitting role for bg-violet-700 #6D28D9; #6D28D9 is accent-strong (a text colour)
          selected && "border-0 bg-violet-700 hover:border",
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
            // TODO(tokens): no fitting role for text-zinc-50 #F9F9FA; #F9F9FA is surface-muted (a surface fill)
            selected && "text-zinc-50",
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
            // TODO(tokens): no fitting role for bg-violet-700 #6D28D9; #6D28D9 is accent-strong (a text colour); no fitting role for text-zinc-50 #F9F9FA; #F9F9FA is surface-muted (a surface fill)
            className="flex h-[1.375rem] items-center rounded-[1.25rem] bg-violet-700 p-[0.375rem] text-zinc-50"
          >
            {number > 100 ? "100+" : number}
          </motion.span>
        )}
      </Button>
    </AnimatePresence>
  );
};
