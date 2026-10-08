"use client";

import { useSearchParams } from "next/navigation";
import { useAuth } from "@/lib/auth/hooks/useAuth";
import { LoadingSpinner } from "@/components/atoms/LoadingSpinner/LoadingSpinner";
import { MobileWorkspace } from "./components/MobileWorkspace";

export default function MobilePage() {
  const { user, isUserLoading } = useAuth();
  const params = useSearchParams();
  if (isUserLoading || !user) return <LoadingSpinner />;
  const tab = params.get("tab");
  return (
    <MobileWorkspace
      tab={tab === "experts" || tab === "attention" ? tab : "chats"}
    />
  );
}
