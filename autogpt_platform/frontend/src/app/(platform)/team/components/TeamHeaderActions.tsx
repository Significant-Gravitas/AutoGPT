"use client";

import { Button } from "@/components/atoms/Button/Button";
import {
  SparklesIcon,
  UserAdd01Icon,
  UserGroupIcon,
} from "@hugeicons/core-free-icons";

interface Props {
  onNewPod: () => void;
}

export function TeamHeaderActions({ onNewPod }: Props) {
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
      <Button
        variant="outline"
        size="xs"
        onClick={onNewPod}
        leadingIcon={UserGroupIcon}
      >
        New Pod
      </Button>
    </div>
  );
}
