"use client";

import { useRef } from "react";

import { usePostV2UploadOrganizationAvatar } from "@/app/api/__generated__/endpoints/orgs/orgs";
import type { OrgResponse } from "@/app/api/__generated__/models/orgResponse";
import { toast } from "@/components/molecules/Toast/use-toast";
import { useOrgTeamStore } from "@/services/org-team/store";

interface Args {
  org: OrgResponse;
  onSaved: () => void;
}

export function useOrgAvatarControl({ org, onSaved }: Args) {
  const fileRef = useRef<HTMLInputElement>(null);
  const { orgs, setOrgs } = useOrgTeamStore();

  const { mutateAsync: uploadAvatar, isPending } =
    usePostV2UploadOrganizationAvatar({
      mutation: {
        onError: (error) => {
          toast({
            title: "Failed to upload avatar",
            description:
              error instanceof Error ? error.message : "Please try again.",
            variant: "destructive",
          });
        },
      },
    });

  function openFilePicker() {
    if (isPending) return;
    fileRef.current?.click();
  }

  async function handleFileChange(event: React.ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    // Reset so re-selecting the same file still fires onChange.
    event.target.value = "";
    if (!file) return;
    try {
      const response = await uploadAvatar({ orgId: org.id, data: { file } });
      const updated = response.data as OrgResponse;
      // Keep the org switcher's avatars in step with the new upload.
      setOrgs(
        orgs.map((o) =>
          o.id === updated.id
            ? { ...o, avatarUrl: updated.avatar_url ?? null }
            : o,
        ),
      );
      toast({ title: "Avatar updated", variant: "success" });
      onSaved();
    } catch {
      // onError already surfaced the failure as a toast.
    }
  }

  return { fileRef, isPending, openFilePicker, handleFileChange };
}
