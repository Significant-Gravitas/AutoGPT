"use client";
import {
  PropsWithChildren,
  RefObject,
  useImperativeHandle,
  useLayoutEffect,
  useRef,
  useState,
} from "react";
import { ProActivationDialog } from "@/components/organisms/ProActivation/ProActivationDialog";
import { ProActivationContext } from "./context";
import { useAuthStore } from "@/lib/auth/hooks/useAuthStore";
import { useActivationController } from "./useActivationController";

interface SessionStatus {
  userID: string | undefined;
  isBusy: boolean;
  isReady: boolean;
}

interface SessionHandle {
  userID: string | undefined;
  start: (returnTo?: string) => Promise<void>;
}

export function ProActivationProvider({ children }: PropsWithChildren) {
  const userID = useAuthStore((state) => state.user?.id);
  const controller = useRef<SessionHandle | null>(null);
  const [status, setStatus] = useState<SessionStatus>();

  async function start(returnTo?: string) {
    if (controller.current?.userID !== userID) return;
    await controller.current?.start(returnTo);
  }

  return (
    <ProActivationContext.Provider
      value={{
        start,
        isBusy: status?.userID === userID && !!status?.isBusy,
        isReady: status?.userID === userID && !!status?.isReady,
      }}
    >
      {children}
      <ActivationSession
        key={userID ?? "signed-out"}
        userID={userID}
        controller={controller}
        onStatus={setStatus}
      />
    </ProActivationContext.Provider>
  );
}

interface SessionProps {
  userID: string | undefined;
  controller: RefObject<SessionHandle | null>;
  onStatus: (status: SessionStatus) => void;
}

function ActivationSession({ userID, controller, onStatus }: SessionProps) {
  const activation = useActivationController();
  useImperativeHandle(controller, () => ({ userID, start: activation.start }));
  useLayoutEffect(() => {
    onStatus({
      userID,
      isBusy: activation.isBusy,
      isReady: activation.isReady,
    });
  }, [userID, activation.isBusy, activation.isReady, onStatus]);
  return <ProActivationDialog activation={activation} />;
}
