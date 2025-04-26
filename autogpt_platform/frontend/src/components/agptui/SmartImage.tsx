"use client";
import { cn } from "@/lib/utils";
import Image from "next/image";
import { useState } from "react";

interface SmartImageProps {
  src?: string | null;
  alt: string;
  className?: string;
}

export default function SmartImage({ src, alt, className }: SmartImageProps) {
  const [isLoading, setIsLoading] = useState(true);
  const shouldShowSkeleton = isLoading || !src;

  return (
    <div className={cn("relative overflow-hidden", className)}>
      {src && (
        <Image
          src={src}
          alt={alt}
          fill
          sizes="100%"
          onLoad={() => setIsLoading(false)}
          className={cn(
            "h-full w-full object-cover object-center transition-opacity duration-200",
            isLoading ? "opacity-0" : "opacity-100",
            "rounded-inherit",
          )}
        />
      )}

      {shouldShowSkeleton && (
        <div className="absolute inset-0 animate-pulse bg-zinc-300" />
      )}
    </div>
  );
}
