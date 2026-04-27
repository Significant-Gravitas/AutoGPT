"use client";

import { useEffect, useMemo, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";

import {
  getGetV2GetUserProfileQueryKey,
  postV2UpdateUserProfile,
  useGetV2GetUserProfile,
  usePostV2UploadSubmissionMedia,
} from "@/app/api/__generated__/endpoints/store/store";
import type { ProfileDetails } from "@/app/api/__generated__/models/profileDetails";
import { toast } from "@/components/molecules/Toast/use-toast";
import { isLogoutInProgress } from "@/lib/autogpt-server-api/helpers";
import { useSupabase } from "@/lib/supabase/hooks/useSupabase";

import {
  isFormDirty,
  profileToFormState,
  validateForm,
  type ProfileFormState,
} from "./helpers";

const EMPTY_FORM: ProfileFormState = {
  name: "",
  username: "",
  description: "",
  avatar_url: "",
  links: ["", "", ""],
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
  const [isSaving, setIsSaving] = useState(false);

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
      const next = [...prev.links];
      next[index] = value;
      return { ...prev, links: next };
    });
  }

  function addLink() {
    setFormState((prev) => ({ ...prev, links: [...prev.links, ""] }));
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
    const res = await uploadMutation.mutateAsync({ data: { file } });
    if (res.status !== 200) return null;
    const url = String(res.data ?? "").trim();
    if (!url) return null;
    patchField("avatar_url", url);
    return url;
  }

  async function saveProfile() {
    if (!validation.valid || isSaving) return;
    setIsSaving(true);
    try {
      const payload = {
        name: formState.name.trim(),
        username: formState.username.trim(),
        description: formState.description.trim(),
        avatar_url: formState.avatar_url,
        links: formState.links.map((l) => l.trim()).filter(Boolean),
      };
      const res = await postV2UpdateUserProfile(payload);
      if (res.status === 200) {
        await queryClient.invalidateQueries({
          queryKey: getGetV2GetUserProfileQueryKey(),
        });
        toast({ title: "Profile saved", variant: "success" });
      } else {
        throw new Error("Update failed");
      }
    } catch (err) {
      toast({
        title: "Failed to save profile",
        description: err instanceof Error ? err.message : undefined,
        variant: "destructive",
      });
    } finally {
      setIsSaving(false);
    }
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
    isSaving,
    dirty,
    validation,
  };
}
