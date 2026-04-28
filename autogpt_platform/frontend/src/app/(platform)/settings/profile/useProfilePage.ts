"use client";

import { useEffect, useMemo, useState } from "react";
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

  const initialState: ProfileFormState = useMemo(
    () =>
      profileQuery.data ? profileToFormState(profileQuery.data) : EMPTY_FORM,
    [profileQuery.data],
  );

  const [formState, setFormState] = useState<ProfileFormState>(EMPTY_FORM);

  useEffect(
    function syncFormStateOnDataLoad() {
      if (profileQuery.data) {
        setFormState(profileToFormState(profileQuery.data));
      }
    },
    [profileQuery.data],
  );

  const validation = useMemo(() => validateForm(formState), [formState]);
  const dirty = useMemo(
    () => isFormDirty(initialState, formState),
    [initialState, formState],
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
    setFormState(initialState);
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
    initialState,
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
