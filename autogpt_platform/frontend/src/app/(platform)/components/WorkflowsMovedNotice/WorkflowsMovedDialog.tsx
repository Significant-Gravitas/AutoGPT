"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import {
  DialogContent,
  DialogRoot,
} from "@/components/molecules/Dialog/Dialog";
import { isComposingEscape } from "@/components/molecules/Dialog/helpers";
import { Cancel01Icon } from "@hugeicons/core-free-icons";
import { useRef } from "react";
import { WorkflowsMovedContent } from "./WorkflowsMovedContent";
import { hasCompetingDialog } from "./helpers";

interface Props {
  isOpen: boolean;
  onDismiss: () => void;
}

export function WorkflowsMovedDialog({ isOpen, onDismiss }: Props) {
  const titleRef = useRef<HTMLHeadingElement>(null);
  return (
    <DialogRoot
      open={isOpen}
      onOpenChange={(open, details) => {
        if (open) return;
        if (isComposingEscape(details)) {
          details.cancel();
          return;
        }
        onDismiss();
      }}
    >
      <DialogContent
        data-workflows-moved-notice=""
        showCloseButton={false}
        initialFocus={titleRef}
        finalFocus={() => !hasCompetingDialog()}
        className="flex max-h-[calc(100dvh-2rem)] max-w-176 flex-col gap-0 overflow-hidden bg-white p-0 sm:max-w-176"
      >
        <Button
          variant="icon"
          size="icon-sm"
          aria-label="Close notice"
          onClick={onDismiss}
          withTooltip={false}
          className="absolute top-4 right-4 z-10 rounded-full border-white/80 bg-white/80"
        >
          <Icon icon={Cancel01Icon} size={16} aria-hidden />
        </Button>
        <WorkflowsMovedContent titleRef={titleRef} onDismiss={onDismiss} />
      </DialogContent>
    </DialogRoot>
  );
}
