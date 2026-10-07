"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";
import { ArrowDown01Icon } from "@hugeicons/core-free-icons";
import { useState } from "react";
import { ExpertSection } from "./ExpertSection";

// Roughly the number of characters that fit in the six-line clamp below.
const CLAMPED_LENGTH = 500;

interface Props {
  text: string;
}

export function ExpertAbout({ text }: Props) {
  const [isExpanded, setIsExpanded] = useState(false);
  const isClampable = text.length > CLAMPED_LENGTH;

  return (
    <ExpertSection title="About">
      <Text
        variant="large"
        tone="secondary"
        unmask={false}
        className={cn(
          "leading-7 whitespace-pre-line",
          isClampable && !isExpanded && "line-clamp-6",
        )}
      >
        {text}
      </Text>
      {isClampable ? (
        <Button
          type="button"
          variant="ghost"
          onClick={() => setIsExpanded((value) => !value)}
          className="mt-2 h-auto min-w-0 gap-1 rounded-none border-0 p-0 text-sm font-medium text-muted-foreground hover:bg-transparent hover:text-zinc-900"
        >
          {isExpanded ? "Show less" : "Read more"}
          <Icon
            icon={ArrowDown01Icon}
            size={14}
            className={cn(
              "transition-transform duration-200",
              isExpanded && "rotate-180",
            )}
          />
        </Button>
      ) : null}
    </ExpertSection>
  );
}
