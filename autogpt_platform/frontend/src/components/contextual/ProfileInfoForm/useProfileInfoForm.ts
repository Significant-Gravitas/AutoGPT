import { useState, type FormEvent } from "react";
import { postV2UpdateUserProfile } from "@/app/api/__generated__/endpoints/store/store";
import { resolveResponse } from "@/app/api/helpers";
import type { ProfileDetails } from "@/app/api/__generated__/models/profileDetails";
import { useToast } from "@/components/molecules/Toast/use-toast";
import {
  isFileTooLarge,
  SUBMISSION_MEDIA_MAX_SIZE_MB,
  uploadSubmissionMediaDirect,
} from "@/lib/direct-upload";

export function useProfileInfoForm(profile: ProfileDetails) {
  const { toast } = useToast();
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [profileData, setProfileData] = useState<ProfileDetails>(profile);

  async function submitForm(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (isSubmitting) return;
    try {
      setIsSubmitting(true);

      const updatedProfile = {
        name: profileData.name,
        username: profileData.username,
        description: profileData.description,
        links: profileData.links.filter((link) => link), // Filter out empty links
        avatar_url: profileData.avatar_url,
      };

      const returnedProfile = await resolveResponse(
        postV2UpdateUserProfile(updatedProfile),
      );
      if (returnedProfile) setProfileData(returnedProfile);
    } catch (error) {
      console.error("Error updating profile:", error);
    } finally {
      setIsSubmitting(false);
    }
  }

  async function handleImageUpload(file: File) {
    if (
      isFileTooLarge({ file, maxSizeMB: SUBMISSION_MEDIA_MAX_SIZE_MB, toast })
    )
      return;

    try {
      const mediaUrl = await uploadSubmissionMediaDirect(file);

      const updatedProfile = {
        ...profileData,
        avatar_url: mediaUrl,
      };

      const returnedProfile = await resolveResponse(
        postV2UpdateUserProfile(updatedProfile),
      );
      if (returnedProfile) setProfileData(returnedProfile);
    } catch (error) {
      toast({
        title: "Failed to upload photo",
        description: error instanceof Error ? error.message : undefined,
        variant: "destructive",
      });
    }
  }

  return {
    profileData,
    setProfileData,
    isSubmitting,
    submitForm,
    handleImageUpload,
  };
}
