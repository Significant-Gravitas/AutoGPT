"use client";

import { LoadingSpinner } from "@/components/atoms/LoadingSpinner/LoadingSpinner";
import { useMountEffect } from "@/hooks/useMountEffect";
import { useRouter } from "next/navigation";

interface Props {
  href: string;
}

export function Redirect({ href }: Props) {
  const router = useRouter();

  useMountEffect(() => {
    router.replace(href);
  });

  return <LoadingSpinner size="large" cover />;
}
