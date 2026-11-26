import { useQuery, useQueryClient } from "@tanstack/react-query";
import { router } from "expo-router";
import { useState } from "react";

import { chatKeys } from "@/lib/api/chat";
import { expertActivityKeys } from "@/lib/api/expertActivity";
import { expertManageKeys, fireExpert, getDetachPreview } from "@/lib/api/expertManage";
import { expertKeys } from "@/lib/api/experts";
import { homeKeys } from "@/lib/api/home";

import { firePauseLine, isGoneError } from "./dangerHelpers";

interface Args {
  expertID: string;
  isOpen: boolean;
  onFired: () => void;
}

export function useFireExpert({ expertID, isOpen, onFired }: Args) {
  const queryClient = useQueryClient();
  const [isFiring, setIsFiring] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [wasOpen, setWasOpen] = useState(isOpen);
  const preview = useQuery({
    queryKey: expertManageKeys.detachPreview(expertID),
    queryFn: () => getDetachPreview(expertID),
    enabled: isOpen,
  });

  if (isOpen !== wasOpen) {
    setWasOpen(isOpen);
    if (isOpen) setError(null);
  }

  function pauseLine() {
    if (preview.isFetching) return "Checking what will pause…";
    if (preview.data) return firePauseLine(preview.data);
    return "We couldn't preview what pauses, but you can still fire them.";
  }

  async function fire() {
    if (isFiring || preview.isFetching) return;
    setIsFiring(true);
    setError(null);
    try {
      await fireExpert(expertID);
    } catch (caught) {
      if (!isGoneError(caught)) {
        setError("Couldn't fire them. Try again.");
        setIsFiring(false);
        return;
      }
    }
    setIsFiring(false);
    void queryClient.invalidateQueries({ queryKey: expertKeys.list });
    void queryClient.invalidateQueries({ queryKey: chatKeys.sessions });
    void queryClient.invalidateQueries({ queryKey: homeKeys.dashboard });
    void queryClient.invalidateQueries({ queryKey: expertActivityKeys.graphSchedules });
    void queryClient.invalidateQueries({ queryKey: expertActivityKeys.followupSchedules });
    void queryClient.invalidateQueries({ queryKey: expertActivityKeys.detail(expertID) });
    onFired();
    router.navigate("/team");
  }

  function confirm() {
    void fire();
  }

  return {
    pauseLine: pauseLine(),
    isPreviewLoading: preview.isFetching,
    isFiring,
    error,
    confirm,
  };
}
