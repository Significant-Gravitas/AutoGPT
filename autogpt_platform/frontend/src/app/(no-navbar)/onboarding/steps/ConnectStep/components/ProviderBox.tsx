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
  "flex w-full flex-col items-center gap-3 rounded-2xl border bg-white px-4 py-6 text-center";

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
      <span className="relative flex h-12 w-12 items-center justify-center">
        <Image
          src={logoSrc}
          alt=""
          width={48}
          height={48}
          className={cn("rounded-xl", state === "coming-soon" && "opacity-50")}
        />
        {isBusy && (
          <Icon
            icon={Loading03Icon}
            size={20}
            className="absolute -bottom-1 -right-1 rounded-full bg-white text-zinc-500 motion-safe:animate-spin"
          />
        )}
      </span>
      <span className="flex flex-col items-center gap-1">
        <Text
          variant="body-medium"
          as="span"
          className={state === "coming-soon" ? "text-zinc-500" : "text-black"}
        >
          {name}
        </Text>
        {state === "connected" && (
          <span className="inline-flex items-center gap-1 rounded-[10px] bg-[#E8F8F0] px-2 py-[2px] text-[13px] font-medium leading-[20px] text-[#157E58]">
            <Icon icon={CheckmarkCircle02Icon} size={13} />
            Connected
          </span>
        )}
        {state === "coming-soon" && (
          <span className="inline-flex items-center rounded-[10px] bg-zinc-100 px-2 py-[2px] text-[13px] font-medium leading-[20px] text-zinc-500">
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
            ? "border-dashed border-zinc-200"
            : "border-zinc-200",
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
        "border-zinc-200 transition-colors hover:border-zinc-300 hover:bg-zinc-50",
        "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-purple-400",
        isBusy && "cursor-progress",
      )}
    >
      {body}
    </button>
  );
}
