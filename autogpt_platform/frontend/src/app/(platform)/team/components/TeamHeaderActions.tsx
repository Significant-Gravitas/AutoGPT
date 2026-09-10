"use client";

import { Button } from "@/components/atoms/Button/Button";
import { SparklesIcon, UserAdd01Icon } from "@hugeicons/core-free-icons";

export function TeamHeaderActions() {
  return (
    <div className="flex flex-wrap items-center gap-2">
      <Button
        as="NextLink"
        href="/raise"
        variant="secondary"
        size="xs"
        leadingIcon={SparklesIcon}
      >
        Raise expert
      </Button>
      <Button
        as="NextLink"
        href="/marketplace#experts"
        variant="primary"
        size="xs"
        leadingIcon={UserAdd01Icon}
      >
        Hire expert
      </Button>
    </div>
  );
}
