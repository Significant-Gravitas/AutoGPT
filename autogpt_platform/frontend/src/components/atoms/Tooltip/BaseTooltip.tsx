"use client";

import {
  Tooltip as KobraTooltip,
  TooltipContent as KobraTooltipContent,
  TooltipProvider as KobraTooltipProvider,
  TooltipTrigger as KobraTooltipTrigger,
} from "@/components/ui/tooltip";
import * as React from "react";

const DEFAULT_DELAY = 10;

const DelayCtx = React.createContext(DEFAULT_DELAY);

interface ProviderProps {
  children: React.ReactNode;
  delayDuration?: number;
  skipDelayDuration?: number;
}

function TooltipProvider({
  children,
  delayDuration,
  skipDelayDuration,
}: ProviderProps) {
  return (
    <KobraTooltipProvider delay={delayDuration} timeout={skipDelayDuration}>
      {children}
    </KobraTooltipProvider>
  );
}

interface Props {
  children: React.ReactNode;
  delayDuration?: number;
  open?: boolean;
  onOpenChange?: (open: boolean) => void;
}

function Tooltip({
  children,
  delayDuration = DEFAULT_DELAY,
  open,
  onOpenChange,
}: Props) {
  return (
    <DelayCtx.Provider value={delayDuration}>
      <KobraTooltip
        open={open}
        onOpenChange={onOpenChange ? (next) => onOpenChange(next) : undefined}
      >
        {children}
      </KobraTooltip>
    </DelayCtx.Provider>
  );
}

// Base UI puts the open delay on the trigger, so the house `Tooltip
// delayDuration` reaches it through context.
function TooltipTrigger({
  delay,
  ...props
}: React.ComponentProps<typeof KobraTooltipTrigger>) {
  const inherited = React.useContext(DelayCtx);
  return <KobraTooltipTrigger delay={delay ?? inherited} {...props} />;
}

// Kobra's content always portals; kept so existing wrappers keep compiling.
function TooltipPortal({ children }: { children: React.ReactNode }) {
  return <>{children}</>;
}

type ContentProps = React.ComponentProps<typeof KobraTooltipContent> & {
  /** Radix collision padding; Base UI positions with its own default. */
  collisionPadding?: number;
};

// Base UI leaves the popup role-less; the house keeps `tooltip` so assistive
// tech and tests can find it as before.
function TooltipContent({ collisionPadding: _, ...props }: ContentProps) {
  return <KobraTooltipContent role="tooltip" {...props} />;
}

export {
  Tooltip,
  TooltipTrigger,
  TooltipContent,
  TooltipPortal,
  TooltipProvider,
};
