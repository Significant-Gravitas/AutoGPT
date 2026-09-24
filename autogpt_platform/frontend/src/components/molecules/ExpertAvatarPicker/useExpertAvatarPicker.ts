import { usePostV2UploadSubmissionMedia } from "@/app/api/__generated__/endpoints/store/store";
import { ExpertAvatarRequestCategory } from "@/app/api/__generated__/models/expertAvatarRequestCategory";
import type { ExpertAvatarRequestShape } from "@/app/api/__generated__/models/expertAvatarRequestShape";
import type { ExpertAvatarRequestExpression } from "@/app/api/__generated__/models/expertAvatarRequestExpression";
import { ACCEPTED_AVATAR_TYPES, MAX_AVATAR_BYTES } from "./helpers";
import {
  EXPERT_AVATARS,
  resolveExpertAvatarUrl,
} from "../ExpertAvatar/helpers";
import { useEffect, useRef, useState } from "react";
import { useAvatarGeneration } from "./useAvatarGeneration";

interface Args {
  avatarUrl?: string | null;
  color: string | null;
  onPick: (url: string, color: string) => void;
}

export function useExpertAvatarPicker({ avatarUrl, color, onPick }: Args) {
  const [selectedUrl, setSelectedUrl] = useState(() =>
    resolveExpertAvatarUrl(avatarUrl),
  );
  const [category, setCategory] = useState<ExpertAvatarRequestCategory>(() => {
    const preset = EXPERT_AVATARS.find(
      (avatar) => avatar.url === resolveExpertAvatarUrl(avatarUrl),
    );
    return (
      Object.values(ExpertAvatarRequestCategory).find(
        (value) => value === preset?.id,
      ) ?? "content"
    );
  });
  const [shape, setShape] = useState<ExpertAvatarRequestShape>("pebble");
  const [expression, setExpression] =
    useState<ExpertAvatarRequestExpression>("friendly");
  const [selectedColor, setSelectedColor] = useState(color ?? "amber-300");
  const [uploadError, setUploadError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const handledJob = useRef("");
  const generation = useAvatarGeneration();
  const upload = usePostV2UploadSubmissionMedia();
  const isBusy = generation.isGenerating || upload.isPending;

  useEffect(() => {
    const job = generation.job;
    if (
      job?.status !== "complete" ||
      !job.id ||
      !job.avatar_url ||
      handledJob.current === job.id
    )
      return;
    handledJob.current = job.id;
    setSelectedUrl(job.avatar_url);
    setSelectedColor(
      EXPERT_AVATARS.find((avatar) => avatar.id === category)?.color ??
        "amber-300",
    );
  }, [generation.job, category]);

  function selectPreset(id: string) {
    const preset = EXPERT_AVATARS.find((avatar) => avatar.id === id);
    const category = Object.values(ExpertAvatarRequestCategory).find(
      (value) => value === id,
    );
    if (!preset || !category) return;
    setCategory(category);
    setSelectedUrl(preset.url);
    setSelectedColor(preset.color);
    setUploadError(null);
  }

  async function uploadFile(file: File | undefined) {
    if (!file) return;
    if (
      !ACCEPTED_AVATAR_TYPES.split(",").includes(file.type) ||
      file.size > MAX_AVATAR_BYTES
    ) {
      setUploadError("Choose a PNG, JPEG, WEBP, or GIF under 5MB.");
      return;
    }
    setUploadError(null);
    try {
      const response = await upload.mutateAsync({ data: { file } });
      if (typeof response.data !== "string" || !response.data.trim())
        throw new Error("No image URL");
      setSelectedUrl(response.data.trim());
    } catch {
      setUploadError(
        "Could not upload that picture. Try again or choose a catalog avatar.",
      );
    }
  }

  function confirm() {
    onPick(selectedUrl, selectedColor);
  }
  function generate() {
    void generation.generate({ category, shape, expression });
  }
  function openFilePicker() {
    fileInputRef.current?.click();
  }

  return {
    selectedUrl,
    category,
    shape,
    setShape,
    expression,
    setExpression,
    selectPreset,
    confirm,
    generate,
    openFilePicker,
    fileInputRef,
    uploadFile,
    isBusy,
    isGenerating: generation.isGenerating,
    error: uploadError ?? generation.error,
  };
}
