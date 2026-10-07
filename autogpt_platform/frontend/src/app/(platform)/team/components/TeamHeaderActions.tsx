"use client";

import { Button } from "@/components/atoms/Button/Button";

export function TeamHeaderActions() {
  return (
    <div className="flex flex-wrap items-center gap-2">
      <Button as="NextLink" href="/raise" variant="secondary" size="md">
        Create an Expert
      </Button>
      <Button
        as="NextLink"
        href="/marketplace#experts"
        variant="primary"
        size="md"
      >
        Hire expert
      </Button>
    </div>
  );
}
