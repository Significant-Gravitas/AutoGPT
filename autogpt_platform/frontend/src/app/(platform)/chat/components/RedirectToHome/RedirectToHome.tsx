"use client";

import { LoadingSpinner } from "@/components/atoms/LoadingSpinner/LoadingSpinner";
import { useMountEffect } from "@/hooks/useMountEffect";
import { useRouter } from "next/navigation";

export function RedirectToHome() {
  const router = useRouter();

  useMountEffect(() => {
    router.replace(`/home${window.location.search}`);
  });

  return <LoadingSpinner size="large" cover />;
}
