"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";

import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";

export default function SettingsIndexPage() {
  const router = useRouter();

  useEffect(() => {
    router.replace("/settings/profile");
  }, [router]);

  return <SettingsContentSkeleton />;
}

function SettingsContentSkeleton() {
  return (
    <div
      className="flex w-full flex-col gap-4 px-4 pt-2"
      aria-busy="true"
      aria-live="polite"
    >
      <Skeleton className="h-7 w-44 rounded-[8px]" />
      <Skeleton className="h-4 w-72 rounded-[8px]" />
      <Skeleton className="mt-2 h-[120px] w-full rounded-[18px]" />
      <Skeleton className="h-[180px] w-full rounded-[18px]" />
      <Skeleton className="h-[120px] w-full rounded-[18px]" />
    </div>
  );
}
