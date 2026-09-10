"use client";

import { Button } from "@/components/atoms/Button/Button";

export function TeamHeaderActions() {
  return (
    <div className="flex flex-wrap items-center gap-2">
      <Button as="NextLink" href="/raise" variant="secondary" size="small">
        Raise expert
      </Button>
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
