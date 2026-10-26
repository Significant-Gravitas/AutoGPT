import type { ExpertAvatarRequestCategory } from "@/app/api/__generated__/models/expertAvatarRequestCategory";
import {
  getFileSizeError,
  SUBMISSION_MEDIA_MAX_SIZE_MB,
  uploadSubmissionMediaDirect,
} from "@/lib/direct-upload";
import { useMutation } from "@tanstack/react-query";
import { useRef, useState } from "react";
import { resolveExpertAvatarUrl } from "../ExpertAvatar/helpers";
import {
  ACCEPTED_AVATAR_TYPES,
  defaultAvatarUrl,
  randomAvatarRequest,
} from "./helpers";
import { useAvatarGeneration } from "./useAvatarGeneration";

interface Args {
  name: string;
  category: ExpertAvatarRequestCategory;
  avatarUrl?: string | null;
  onPick: (url: string) => void;
}

export function useExpertAvatarPicker({
  name,
  category,
  avatarUrl,
  onPick,
}: Args) {
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const generation = useAvatarGeneration();
  const upload = useMutation({
    mutationFn: (file: File) =>
      uploadSubmissionMediaDirect(file, "expert-avatar"),
  });

  const generatedUrl =
    generation.job?.status === "complete" ? generation.job.avatar_url : null;
  const selectedUrl =
    generatedUrl ??
    previewUrl ??
    (avatarUrl
      ? resolveExpertAvatarUrl(avatarUrl)
      : defaultAvatarUrl(category, name));

  function generate() {
    setPreviewUrl(selectedUrl);
    setUploadError(null);
    generation.generate(randomAvatarRequest(category));
  }

  async function uploadFile(file: File | undefined) {
    if (!file) return;
    const sizeError = getFileSizeError(file, SUBMISSION_MEDIA_MAX_SIZE_MB);
    if (!ACCEPTED_AVATAR_TYPES.split(",").includes(file.type) || sizeError) {
      setUploadError(sizeError ?? "Choose a PNG, JPEG, or WEBP.");
      return;
    }
    setUploadError(null);
    setPreviewUrl(selectedUrl);
    generation.reset();
    try {
      const response = await upload.mutateAsync(file);
      if (typeof response !== "string" || !response.trim())
        throw new Error("No image URL");
      setPreviewUrl(response.trim());
    } catch {
      setUploadError("Could not upload that picture. Try again or regenerate.");
    }
  }

  function confirm() {
    setPreviewUrl(selectedUrl);
    generation.reset();
    onPick(selectedUrl);
  }

  function openFilePicker() {
    fileInputRef.current?.click();
  }

  return {
    selectedUrl,
    confirm,
    generate,
    openFilePicker,
    fileInputRef,
    uploadFile,
    isBusy: generation.isGenerating || upload.isPending,
    isUploading: upload.isPending,
    isGenerating: generation.isGenerating,
    error: uploadError ?? generation.error,
  };
}
