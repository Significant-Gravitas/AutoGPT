import type { ExpertAvatarRequestCategory } from "@/app/api/__generated__/models/expertAvatarRequestCategory";
import { useMountEffect } from "@/hooks/useMountEffect";
import { uploadSubmissionMediaDirect } from "@/lib/direct-upload";
import { useMutation } from "@tanstack/react-query";
import { useRef, useState } from "react";
import {
  getCategoryAvatarUrl,
  resolveCategoryAvatarUrl,
} from "../ExpertAvatar/helpers";
import {
  ACCEPTED_AVATAR_TYPES,
  MAX_AVATAR_BYTES,
  randomAvatarRequest,
} from "./helpers";
import { useAvatarGeneration } from "./useAvatarGeneration";

interface Args {
  category: ExpertAvatarRequestCategory;
  avatarUrl?: string | null;
  autoGenerate?: boolean;
  onPick: (url: string) => void;
}

export function useExpertAvatarPicker({
  category,
  avatarUrl,
  autoGenerate,
  onPick,
}: Args) {
  const [uploadedUrl, setUploadedUrl] = useState<string | null>(null);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const generation = useAvatarGeneration();
  const upload = useMutation({
    mutationFn: (file: File) =>
      uploadSubmissionMediaDirect(file, "expert-avatar"),
  });

  // Generating resets the upload and uploading resets the generation, so the
  // two never both hold a result and the preview needs no synchronising.
  const generatedUrl =
    generation.job?.status === "complete" ? generation.job.avatar_url : null;
  const selectedUrl =
    generatedUrl ??
    uploadedUrl ??
    (avatarUrl
      ? resolveCategoryAvatarUrl(avatarUrl)
      : getCategoryAvatarUrl(category));

  useMountEffect(() => {
    if (autoGenerate) generate();
  });

  function generate() {
    setUploadedUrl(null);
    setUploadError(null);
    generation.generate(randomAvatarRequest(category));
  }

  async function uploadFile(file: File | undefined) {
    if (!file) return;
    if (
      !ACCEPTED_AVATAR_TYPES.split(",").includes(file.type) ||
      file.size > MAX_AVATAR_BYTES
    ) {
      setUploadError("Choose a PNG, JPEG, or WEBP under 5MB.");
      return;
    }
    setUploadError(null);
    try {
      const response = await upload.mutateAsync(file);
      if (typeof response !== "string" || !response.trim())
        throw new Error("No image URL");
      generation.reset();
      setUploadedUrl(response.trim());
    } catch {
      setUploadError("Could not upload that picture. Try again or regenerate.");
    }
  }

  function confirm() {
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
    isGenerating: generation.isGenerating,
    error: uploadError ?? generation.error,
  };
}
