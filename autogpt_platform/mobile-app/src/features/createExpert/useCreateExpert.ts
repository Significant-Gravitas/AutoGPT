import { router } from "expo-router";
import { useRef, useState } from "react";

import { avatarSource } from "@/features/team/helpers";
import { chatKeys, createSession } from "@/lib/api/chat";
import { createExpert, expertKeys } from "@/lib/api/experts";
import { expertTint } from "@/lib/expertPalette";
import { isSessionGone, queryClient, signOut } from "@/lib/session";

import {
  canCreate,
  createErrorMessage,
  createExpertBody,
  DEFAULT_FAMILY,
  defaultAvatarPath,
  type Family,
} from "./helpers";

export function useCreateExpert() {
  const [name, setName] = useState("");
  const [jobTitle, setJobTitle] = useState("");
  const [family, setFamily] = useState<Family>(DEFAULT_FAMILY);
  const [isCreating, setIsCreating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  // State lands a render late; the ref stops a second tap in the same frame.
  const isCreatingRef = useRef(false);
  const avatarPath = defaultAvatarPath(family, name);

  function close() {
    if (router.canGoBack()) router.back();
    else router.navigate("/team");
  }

  async function openChatWith(expertID: string | undefined) {
    if (!expertID) {
      router.replace("/team");
      return;
    }
    const { id } = await createSession(expertID);
    void queryClient.invalidateQueries({ queryKey: chatKeys.sessions });
    router.replace(`/chat/${id}`);
  }

  async function create() {
    if (isCreatingRef.current || !canCreate(name)) return;
    isCreatingRef.current = true;
    setIsCreating(true);
    setError(null);
    let expertID: string | undefined;
    try {
      const result = await createExpert(createExpertBody({ name, jobTitle, family }));
      expertID = result.expert?.id;
      await queryClient.invalidateQueries({ queryKey: expertKeys.list });
      await openChatWith(expertID);
    } catch (caught) {
      if (isSessionGone(caught)) {
        void signOut();
      } else if (expertID) {
        // The expert exists; only its first chat failed to open.
        router.replace("/team");
      } else {
        setError(createErrorMessage(caught));
      }
    } finally {
      isCreatingRef.current = false;
      setIsCreating(false);
    }
  }

  return {
    name,
    setName,
    jobTitle,
    setJobTitle,
    family,
    setFamily,
    previewSource: avatarSource({ avatar_url: avatarPath }),
    previewTint: expertTint({ avatarUrl: avatarPath, categories: [family] }),
    canCreate: canCreate(name),
    isCreating,
    error,
    create,
    close,
  };
}
