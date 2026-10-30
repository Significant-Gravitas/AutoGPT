"use client";

import { Button } from "@/components/atoms/Button/Button";
import { RaiseExpertButton } from "./RaiseExpertButton/RaiseExpertButton";

export function TeamHeaderActions() {
  return (
    <div className="flex flex-wrap items-center gap-2">
      <RaiseExpertButton />
      <Button
        as="NextLink"
        href="/marketplace#experts"
        variant="primary"
        size="small"
      >
        Hire expert
      </Button>
    </div>
  );
}
