"use client";

import React, {
  createContext,
  useContext,
  useEffect,
  useMemo,
  useState,
} from "react";
import BoringAvatar from "boring-avatars";

import Image, { ImageProps } from "next/image";
import { cn } from "@/lib/utils";

type AvatarContextValue = {
  isLoaded: boolean;
  setIsLoaded: (v: boolean) => void;
  hasImage: boolean;
  setHasImage: (v: boolean) => void;
};

const AvatarContext = createContext<AvatarContextValue | null>(null);

function useAvatarContext(): AvatarContextValue {
  const ctx = useContext(AvatarContext);
  if (!ctx) throw new Error("Avatar components must be used within <Avatar />");
  return ctx;
}

export type AvatarProps = React.HTMLAttributes<HTMLDivElement>;

export function Avatar({
  className,
  children,
  ...props
}: AvatarProps): JSX.Element {
  const [isLoaded, setIsLoaded] = useState<boolean>(false);
  const [hasImage, setHasImage] = useState<boolean>(false);

  const value = useMemo(
    () => ({ isLoaded, setIsLoaded, hasImage, setHasImage }),
    [isLoaded, hasImage],
  );

  return (
    <AvatarContext.Provider value={value}>
      <div
        className={cn(
          "relative flex h-10 w-10 shrink-0 overflow-hidden rounded-full",
          className,
        )}
        {...props}
      >
        {children}
      </div>
    </AvatarContext.Provider>
  );
}

export interface AvatarImageProps
  extends Omit<React.ImgHTMLAttributes<HTMLImageElement>, "width" | "height"> {
  as?: "NextImage" | "img";
  width?: number;
  height?: number;
  fill?: boolean;
  sizes?: string;
  priority?: boolean;
  unoptimized?: boolean;
}

function getAvatarSizeFromClassName(className?: string): number | null {
  if (!className) return null;
  // Try to parse classes like h-16 w-16 first
  const hMatch = className.match(/\bh-(\d+)\b/);
  if (hMatch) return parseInt(hMatch[1], 10) * 4; // Tailwind spacing scale default: 1 => 0.25rem
  const wMatch = className.match(/\bw-(\d+)\b/);
  if (wMatch) return parseInt(wMatch[1], 10) * 4;
  // Fallback fixed size
  return null;
}

export function AvatarImage({
  as = "NextImage",
  src,
  alt,
  className,
  onLoad,
  onError,
  width,
  height,
  fill,
  sizes,
  priority,
  unoptimized,
  ...rest
}: AvatarImageProps): JSX.Element | null {
  const { setIsLoaded, setHasImage, hasImage } = useAvatarContext();

  const normalizedSrc = typeof src === "string" ? src.trim() : src;

  useEffect(
    function setHasImageOnSrcChange() {
      setHasImage(Boolean(normalizedSrc));
    },
    [normalizedSrc, setHasImage],
  );

  if (!normalizedSrc || !hasImage) return null;

  const sizeFromClass = getAvatarSizeFromClassName(className);
  const computedWidth = width || sizeFromClass || 40;
  const computedHeight = height || sizeFromClass || 40;

  if (as === "img") {
    function handleLoad(e: React.SyntheticEvent<HTMLImageElement, Event>) {
      setIsLoaded(true);
      if (onLoad) onLoad(e);
    }
    function handleError(e: React.SyntheticEvent<HTMLImageElement, Event>) {
      setIsLoaded(false);
      setHasImage(false);
      if (onError) onError(e);
    }

    return (
      // eslint-disable-next-line @next/next/no-img-element
      <img
        src={normalizedSrc}
        alt={alt || "Avatar image"}
        className={cn("h-full w-full object-cover", className)}
        width={computedWidth}
        height={computedHeight}
        onLoad={handleLoad}
        onError={handleError}
        {...rest}
      />
    );
  }

  function handleLoadingComplete(): void {
    setIsLoaded(true);
  }
  function handleErrorNext(): void {
    setIsLoaded(false);
    setHasImage(false);
  }

  return (
    <Image
      src={normalizedSrc}
      alt={alt || "Avatar image"}
      className={cn("h-full w-full object-cover", className)}
      width={fill ? undefined : computedWidth}
      height={fill ? undefined : computedHeight}
      fill={Boolean(fill)}
      sizes={sizes}
      priority={priority}
      unoptimized={unoptimized}
      onLoadingComplete={handleLoadingComplete}
      onError={handleErrorNext as ImageProps["onError"]}
    />
  );
}

export type AvatarFallbackProps = React.HTMLAttributes<HTMLSpanElement> & {
  size?: number;
};

export function AvatarFallback({
  className,
  children,
  size: _size, // accepted for API compatibility; currently not used
  ...props
}: AvatarFallbackProps): JSX.Element | null {
  const { isLoaded, hasImage } = useAvatarContext();
  const show = !isLoaded || !hasImage;
  if (!show) return null;
  const computedSize = _size || getAvatarSizeFromClassName(className) || 40;
  const name =
    typeof children === "string" && children.trim() ? children : "User";
  return (
    <span
      className={cn(
        "flex h-full w-full items-center justify-center rounded-full bg-transparent text-lg text-neutral-600",
        className,
      )}
      {...props}
    >
      <BoringAvatar
        size={computedSize}
        name={name}
        variant="marble"
        colors={["#92A1C6", "#146A7C", "#F0AB3D", "#C271B4", "#C20D90"]}
        square={false}
      />
    </span>
  );
}

export default Avatar;
