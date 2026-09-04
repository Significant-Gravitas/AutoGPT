"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { SparklesIcon, UserGroupIcon } from "@hugeicons/core-free-icons";
import { ACTION_BUTTON_CLASS, OUTLINE_ACTION_BUTTON_CLASS } from "../helpers";

interface Props {
  onNewPod: () => void;
}

export function TeamHeaderActions({ onNewPod }: Props) {
  return (
    <div className="flex flex-wrap items-center gap-2">
      <Button
        as="NextLink"
        href="/raise"
        variant="primary"
        size="small"
        className={ACTION_BUTTON_CLASS}
        leftIcon={<Icon icon={SparklesIcon} className="size-4" />}
      >
        New Expert
      </Button>
      <Button
        variant="outline"
        size="small"
        onClick={onNewPod}
        className={OUTLINE_ACTION_BUTTON_CLASS}
        leftIcon={<Icon icon={UserGroupIcon} className="size-4" />}
      >
        New Pod
      </Button>
    </div>
  );
}
