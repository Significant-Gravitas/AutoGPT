"use client";

import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";
import {
  CheckmarkCircle02Icon,
  Loading03Icon,
} from "@hugeicons/core-free-icons";
import Image from "next/image";

interface Props {
  name: string;
  logoSrc: string;
  state: "available" | "connected" | "coming-soon";
  isBusy?: boolean;
  onClick?: () => void;
}

const FRAME =
  "flex h-44 w-full min-w-0 flex-col items-center gap-3 rounded-2xl border bg-surface px-3 py-4 text-center";

// One subscription the experts could run on. Available ones are a button
// that starts the sign-in; connected and coming-soon ones just say so.
export function ProviderBox({
  name,
  logoSrc,
  state,
  isBusy = false,
  onClick,
}: Props) {
  const body = (
    <>
      <span className="relative flex h-12 w-12 shrink-0 items-center justify-center">
        <Image
          src={logoSrc}
          alt=""
          width={48}
          height={48}
          className={cn(
            "h-12 w-12 rounded-xl object-contain",
            state === "coming-soon" && "opacity-50",
          )}
        />
        {isBusy && (
          <Icon
            icon={Loading03Icon}
            size={20}
            className="absolute -bottom-1 -right-1 rounded-full bg-surface text-foreground-tertiary motion-safe:animate-spin"
          />
        )}
      </span>
      <span className="flex w-full flex-col items-center gap-1">
        <Text
          variant="body-medium"
          as="span"
          className={cn(
            "flex min-h-11 items-center justify-center",
            state === "coming-soon"
              ? "text-foreground-tertiary"
              : "text-foreground-base",
          )}
        >
          {name}
        </Text>
        {state === "connected" && (
          <span
            // TODO(tokens): no exact role for bg-emerald-50 #ecfdf5, text-emerald-700 #047857
            className="inline-flex items-center gap-1 rounded-[10px] bg-emerald-50 px-2 py-[2px] text-body-sm font-medium leading-5 text-emerald-700"
          >
            <Icon icon={CheckmarkCircle02Icon} size={13} />
            Connected
          </span>
        )}
        {state === "coming-soon" && (
          <span className="inline-flex items-center rounded-[10px] bg-surface-hover px-2 py-[2px] text-body-sm font-medium leading-5 text-foreground-tertiary">
            Coming soon
          </span>
        )}
      </span>
    </>
  );

  if (state !== "available") {
    return (
      <div
        className={cn(
          FRAME,
          state === "coming-soon"
            ? "border-dashed border-border-base"
            : "border-border-base",
        )}
      >
        {body}
      </div>
    );
  }

  return (
    <button
      type="button"
      onClick={onClick}
      aria-busy={isBusy}
      disabled={isBusy}
      className={cn(
        FRAME,
        "border-border-base transition-colors hover:border-border-strong hover:bg-surface-muted",
        // TODO(tokens): no exact role for focus-visible:ring-purple-400 #925cf7
        "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-purple-400",
        isBusy && "cursor-progress",
      )}
    >
      {body}
    </button>
  );
}
