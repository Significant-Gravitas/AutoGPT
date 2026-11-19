"use client";

import { Icon } from "@/components/atoms/Icon/Icon";
import { cn } from "@/lib/utils";
import { Copy01Icon, Tick02Icon } from "@hugeicons/core-free-icons";
import type { IconSvgElement } from "@hugeicons/react";
import { useEffect, useState } from "react";
import { FOCUS } from "../EmptySession/components/ExpertInfoPanel/components/InfoParts";

const COPIED_MS = 1500;

interface ToolProps {
  icon: IconSvgElement;
  label: string;
  /** Shows the label beside the icon, where there is room for it. */
  isLabelled?: boolean;
  onClick: () => void;
}

/** One thing you can do with the open file. */
export function Tool({ icon, label, isLabelled, onClick }: ToolProps) {
  return (
    <button
      type="button"
      aria-label={isLabelled ? undefined : label}
      title={isLabelled ? undefined : label}
      onClick={onClick}
      className={cn(
        "flex h-7 shrink-0 items-center justify-center gap-1.5 rounded-md text-body-sm font-medium text-foreground-quiet transition-colors duration-100 ease-out hover:bg-foreground-ink/5 hover:text-foreground-ink",
        isLabelled ? "px-2" : "w-7",
        FOCUS,
      )}
    >
      <Icon icon={icon} size={14} tone="inherit" />
      {isLabelled ? label : null}
    </button>
  );
}

interface CopyProps {
  isLabelled?: boolean;
  onCopy: () => void;
}

/** Copy answers back: its icon turns to a check for a moment. */
export function CopyTool({ isLabelled, onCopy }: CopyProps) {
  const [isCopied, setIsCopied] = useState(false);
  useEffect(() => {
    if (!isCopied) return;
    const timer = window.setTimeout(() => setIsCopied(false), COPIED_MS);
    return () => window.clearTimeout(timer);
  }, [isCopied]);

  return (
    <Tool
      icon={isCopied ? Tick02Icon : Copy01Icon}
      label={isCopied ? "Copied" : "Copy"}
      isLabelled={isLabelled}
      onClick={() => {
        onCopy();
        setIsCopied(true);
      }}
    />
  );
}

interface SwitchProps {
  isSource: boolean;
  onChange: (isSource: boolean) => void;
}

/** Preview or Source: the one in view sits on a raised pill. */
export function ViewSwitch({ isSource, onChange }: SwitchProps) {
  return (
    <div className="flex h-7 shrink-0 items-center rounded-md bg-foreground-ink/5 p-0.5">
      {[
        { label: "Preview", value: false },
        { label: "Source", value: true },
      ].map((option) => (
        <button
          key={option.label}
          type="button"
          aria-pressed={isSource === option.value}
          onClick={() => onChange(option.value)}
          className={cn(
            "flex h-6 items-center rounded-sm px-2 text-caption font-medium transition-colors duration-100 ease-out",
            isSource === option.value
              ? "bg-surface text-foreground-ink shadow-(--inbox-shadow-card)"
              : "text-foreground-quiet hover:text-foreground-ink",
            FOCUS,
          )}
        >
          {option.label}
        </button>
      ))}
    </div>
  );
}
