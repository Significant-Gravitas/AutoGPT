"use client";
import { cn } from "@/lib/utils";
import Image from "next/image";
import { useState } from "react";

interface SmartImageProps {
  src?: string | null;
  alt: string;
  imageContain?: boolean;
  className?: string;
}

export default function SmartImage({
  src,
  alt,
  imageContain,
  className,
}: SmartImageProps) {
  const [isLoading, setIsLoading] = useState(true);
  const shouldShowSkeleton = isLoading || !src;

  return (
    <div className={cn("relative overflow-hidden", className)}>
      {src && (
        <Image
          src={src}
          alt={alt}
          fill
          onLoad={() => setIsLoading(false)}
          className={cn(
            "rounded-inherit object-center transition-opacity duration-300",
            isLoading ? "opacity-0" : "opacity-100",
            imageContain ? "object-contain" : "object-cover",
          )}
          sizes="100%"
        />
      )}
      {shouldShowSkeleton && (
        <div className="rounded-inherit absolute inset-0 animate-pulse bg-gray-300 dark:bg-gray-700" />
      )}
    </div>
  );
}
