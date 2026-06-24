"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";

import {
  getGetV2GetUserProfileQueryKey,
  useGetV2GetUserProfile,
  usePostV2UpdateUserProfile,
  usePostV2UploadSubmissionMedia,
} from "@/app/api/__generated__/endpoints/store/store";
import type { ProfileDetails } from "@/app/api/__generated__/models/profileDetails";
import { toast } from "@/components/molecules/Toast/use-toast";
import { isLogoutInProgress } from "@/lib/autogpt-server-api/helpers";
import { useSupabase } from "@/lib/supabase/hooks/useSupabase";

import {
  isFormDirty,
  makeLinkRow,
  MAX_LINKS,
  profileToFormState,
  validateForm,
  type ProfileFormState,
} from "./helpers";

const EMPTY_FORM: ProfileFormState = {
  name: "",
  username: "",
  description: "",
  avatar_url: "",
  links: [],
};

export function useProfilePage() {
  const { user } = useSupabase();
  const queryClient = useQueryClient();
  const logoutInProgress = isLogoutInProgress();

  const profileQuery = useGetV2GetUserProfile<ProfileDetails | null>({
    query: {
      enabled: !!user && !logoutInProgress,
      select: (res) => {
        if (res.status === 200) {
          return { ...res.data, avatar_url: res.data.avatar_url ?? "" };
        }
        return null;
      },
    },
  });

  const [formState, setFormState] = useState<ProfileFormState>(EMPTY_FORM);
  // Pristine baseline for dirty detection + Discard. Held as state (not a
  // ref) so dirty re-derives when it changes; pinned to the current
  // formState's link IDs on every sync so Discard never swaps IDs and
  // triggers an AnimatePresence flash.
  const [pristineState, setPristineState] =
    useState<ProfileFormState>(EMPTY_FORM);
  const formStateRef = useRef<ProfileFormState>(EMPTY_FORM);
  const lastSyncedRef = useRef<ProfileFormState>(EMPTY_FORM);
  const hasHydratedRef = useRef(false);

  useEffect(() => {
    formStateRef.current = formState;
  }, [formState]);

  useEffect(
    function syncFormStateOnDataLoad() {
      if (!profileQuery.data) return;
      const incoming = profileToFormState(profileQuery.data);
      const prev = formStateRef.current;

      // First load always hydrates — even when the server returns an empty
      // profile, we need the padded link slots in formState to render.
      if (!hasHydratedRef.current) {
        hasHydratedRef.current = true;
        lastSyncedRef.current = incoming;
        setFormState(incoming);
        setPristineState(incoming);
        return;
      }
      // Refetch returned the same content as the local form — keep prev to
      // preserve link IDs (avoids a row exit/enter flash) and refresh the
      // synced snapshot so future dirty checks compare against the latest.
      if (!isFormDirty(incoming, prev)) {
        lastSyncedRef.current = incoming;
        // Pin the pristine baseline to the current formState (with its
        // existing link IDs) so a subsequent Discard restores the same
        // rows — not freshly-keyed copies that would re-trigger animations.
        setPristineState(prev);
        return;
      }
      // User has edits relative to the last synced snapshot — never clobber.
      if (isFormDirty(lastSyncedRef.current, prev)) return;
      // Form pristine, server data changed — hydrate with the new snapshot.
      lastSyncedRef.current = incoming;
      setFormState(incoming);
      setPristineState(incoming);
    },
    [profileQuery.data],
  );

  const validation = useMemo(() => validateForm(formState), [formState]);
  const dirty = useMemo(
    () => isFormDirty(pristineState, formState),
    [pristineState, formState],
  );

  function patchField<K extends keyof ProfileFormState>(
    key: K,
    value: ProfileFormState[K],
  ) {
    setFormState((prev) => ({ ...prev, [key]: value }));
  }

  function setLink(index: number, value: string) {
    setFormState((prev) => {
      const next = prev.links.map((row, i) =>
        i === index ? { ...row, value } : row,
      );
      return { ...prev, links: next };
    });
  }

  function addLink() {
    setFormState((prev) =>
      prev.links.length >= MAX_LINKS
        ? prev
        : { ...prev, links: [...prev.links, makeLinkRow("")] },
    );
  }

  function removeLink(index: number) {
    setFormState((prev) => ({
      ...prev,
      links: prev.links.filter((_, i) => i !== index),
    }));
  }

  function discardChanges() {
    setFormState(pristineState);
  }

  const uploadMutation = usePostV2UploadSubmissionMedia({
    mutation: {
      onError: (err) => {
        toast({
          title: "Failed to upload photo",
          description: err instanceof Error ? err.message : undefined,
          variant: "destructive",
        });
      },
    },
  });

  async function uploadAvatar(file: File): Promise<string | null> {
    try {
      const res = await uploadMutation.mutateAsync({ data: { file } });
      if (res.status !== 200) return null;
      const url = String(res.data ?? "").trim();
      if (!url) return null;
      patchField("avatar_url", url);
      return url;
    } catch {
      return null;
    }
  }

  const updateMutation = usePostV2UpdateUserProfile({
    mutation: {
      onSuccess: () => {
        void queryClient.invalidateQueries({
          queryKey: getGetV2GetUserProfileQueryKey(),
        });
        toast({ title: "Profile saved", variant: "success" });
      },
      onError: (err) => {
        toast({
          title: "Failed to save profile",
          description: err instanceof Error ? err.message : undefined,
          variant: "destructive",
        });
      },
    },
  });

  function saveProfile() {
    if (!validation.valid || updateMutation.isPending) return;
    updateMutation.mutate({
      data: {
        name: formState.name.trim(),
        username: formState.username.trim(),
        description: formState.description.trim(),
        avatar_url: formState.avatar_url,
        links: formState.links
          .map((l) => l.value.trim())
          .filter(Boolean)
          .slice(0, MAX_LINKS),
      },
    });
  }

  return {
    user,
    isLoading: profileQuery.isLoading,
    isError: profileQuery.isError,
    error: profileQuery.error,
    refetch: profileQuery.refetch,
    formState,
    patchField,
    setLink,
    addLink,
    removeLink,
    discardChanges,
    uploadAvatar,
    isUploading: uploadMutation.isPending,
    saveProfile,
    isSaving: updateMutation.isPending,
    dirty,
    validation,
  };
}
