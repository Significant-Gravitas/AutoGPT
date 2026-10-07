"use client";
import { PropsWithChildren } from "react";
import { ProActivationDialog } from "@/components/organisms/ProActivation/ProActivationDialog";
import { ProActivationContext } from "./context";
import { useAuthStore } from "@/lib/auth/hooks/useAuthStore";
import { useActivationController } from "./useActivationController";

export function ProActivationProvider({ children }: PropsWithChildren) {
  const userID = useAuthStore((state) => state.user?.id);
  return (
    <ActivationSession key={userID ?? "signed-out"}>
      {children}
    </ActivationSession>
  );
}

function ActivationSession({ children }: PropsWithChildren) {
  const activation = useActivationController();
  return (
    <ProActivationContext.Provider
      value={{
        start: activation.start,
        isBusy: activation.isBusy,
        isReady: activation.isReady,
      }}
    >
      {children}
      <ProActivationDialog activation={activation} />
    </ProActivationContext.Provider>
  );
}
