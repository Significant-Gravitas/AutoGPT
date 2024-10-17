"use client";

import * as React from "react";
import * as AvatarPrimitive from "@radix-ui/react-avatar";
import BoringAvatar from "./BoringAvatarWrapper";
import tailwindConfig from "../../../tailwind.config";

import { cn } from "@/lib/utils";

const Avatar = React.forwardRef<
  React.ElementRef<typeof AvatarPrimitive.Root>,
  React.ComponentPropsWithoutRef<typeof AvatarPrimitive.Root>
>(({ className, ...props }, ref) => (
  <AvatarPrimitive.Root
    ref={ref}
    className={cn(
      "relative flex h-10 w-10 shrink-0 overflow-hidden rounded-full",
      className,
    )}
    {...props}
  />
));
Avatar.displayName = AvatarPrimitive.Root.displayName;

const AvatarImage = React.forwardRef<
  React.ElementRef<typeof AvatarPrimitive.Image>,
  React.ComponentPropsWithoutRef<typeof AvatarPrimitive.Image>
>(({ className, ...props }, ref) => (
  <AvatarPrimitive.Image
    ref={ref}
    className={cn("aspect-square h-full w-full", className)}
    {...props}
  />
));
AvatarImage.displayName = AvatarPrimitive.Image.displayName;

/**
 * Hack to match the avatar size based on Tailwind classes.
 * This function attempts to extract the size from a 'h-' class in the className string,
 * and maps it to the corresponding size in the Tailwind config.
 * If no matching class is found, it defaults to 40.
 * @param className - The className string to parse
 * @returns The size of the avatar in pixels
 */
const getAvatarSize = (className: string | undefined): number => {
  if (className?.includes("h-")) {
    const match = parseInt(className.match(/h-(\d+)/)?.[1] || "16");
    if (match) {
      const size =
        tailwindConfig.theme.extend.spacing[
          match as keyof typeof tailwindConfig.theme.extend.spacing
        ];
      return size ? parseInt(size.replace("rem", "")) * 16 : 40;
    }
  }
  return 40;
};

const AvatarFallback = React.forwardRef<
  React.ElementRef<typeof AvatarPrimitive.Fallback>,
  React.ComponentPropsWithoutRef<typeof AvatarPrimitive.Fallback>
>(({ className, ...props }, ref) => (
  <AvatarPrimitive.Fallback
    ref={ref}
    className={cn(
      "flex h-full w-full items-center justify-center rounded-full",
      className,
    )}
    {...props}
  >
    <BoringAvatar
      size={getAvatarSize(className)}
      name={props.children?.toString() || "User"}
      variant="marble"
      colors={["#92A1C6", "#146A7C", "#F0AB3D", "#C271B4", "#C20D90"]}
    />
  </AvatarPrimitive.Fallback>
));
AvatarFallback.displayName = AvatarPrimitive.Fallback.displayName;

export { Avatar as Avatar, AvatarImage, AvatarFallback };
