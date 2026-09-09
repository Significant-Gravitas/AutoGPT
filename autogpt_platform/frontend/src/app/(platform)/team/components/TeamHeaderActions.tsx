"use client";

import { Button } from "@/components/atoms/Button/Button";

interface Props {
  onNewPod: () => void;
}

export function TeamHeaderActions({ onNewPod }: Props) {
  return (
    <div className="flex flex-wrap items-center gap-2">
      <Button as="NextLink" href="/raise" variant="secondary" size="small">
        Raise expert
      </Button>
      <Button variant="secondary" size="small" onClick={onNewPod}>
        New Pod
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
