import type { ExpertAvatarRequestAccentPlacement } from "@/app/api/__generated__/models/expertAvatarRequestAccentPlacement";
import type { ExpertAvatarRequestAccentCount } from "@/app/api/__generated__/models/expertAvatarRequestAccentCount";
import { uploadSubmissionMediaDirect } from "@/lib/direct-upload";
import { useMutation } from "@tanstack/react-query";
import { ExpertAvatarRequestCategory } from "@/app/api/__generated__/models/expertAvatarRequestCategory";
import type { ExpertAvatarRequestShade } from "@/app/api/__generated__/models/expertAvatarRequestShade";
import type { ExpertAvatarRequestBase } from "@/app/api/__generated__/models/expertAvatarRequestBase";
import type { ExpertAvatarRequestTilt } from "@/app/api/__generated__/models/expertAvatarRequestTilt";
import type { ExpertAvatarRequestInlay } from "@/app/api/__generated__/models/expertAvatarRequestInlay";
import type { ExpertAvatarRequestShape } from "@/app/api/__generated__/models/expertAvatarRequestShape";
import type { ExpertAvatarRequestExpression } from "@/app/api/__generated__/models/expertAvatarRequestExpression";
import { ACCEPTED_AVATAR_TYPES, MAX_AVATAR_BYTES } from "./helpers";
import {
  EXPERT_AVATARS,
  BUILTIN_EXPERT_AVATARS,
  resolveCategoryAvatarUrl,
  getManagedAvatar,
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
    resolveCategoryAvatarUrl(avatarUrl),
  );
  const [category, setCategory] = useState<ExpertAvatarRequestCategory>(() => {
    const preset = EXPERT_AVATARS.find(
      (avatar) => avatar.url === resolveCategoryAvatarUrl(avatarUrl),
    );
    const managed = getManagedAvatar(resolveCategoryAvatarUrl(avatarUrl), 128);
    if (managed?.assetID === "expert-mina") return "finance";
    if (managed?.assetID === "expert-maria") return "marketing";
    const builtin = BUILTIN_EXPERT_AVATARS.find(
      (avatar) => avatar.url === resolveCategoryAvatarUrl(avatarUrl),
    );
    return (
      Object.values(ExpertAvatarRequestCategory).find(
        (value) => value === (builtin?.primary_category ?? preset?.id),
      ) ?? "content"
    );
  });
  const [shade, setShade] =
    useState<NonNullable<ExpertAvatarRequestShade>>("standard");
  const [base, setBase] = useState<ExpertAvatarRequestBase>("compact");
  const [tilt, setTilt] = useState<ExpertAvatarRequestTilt>("level");
  const [inlay, setInlay] = useState<ExpertAvatarRequestInlay>("sweep");
  const [accentPlacement, setAccentPlacement] =
    useState<ExpertAvatarRequestAccentPlacement>("body");
  const [accentCount, setAccentCount] =
    useState<ExpertAvatarRequestAccentCount>("one");
  const [shape, setShape] = useState<ExpertAvatarRequestShape>("pebble");
  const [expression, setExpression] =
    useState<ExpertAvatarRequestExpression>("friendly");
  const [selectedColor, setSelectedColor] = useState(color ?? "amber-300");
  const [uploadError, setUploadError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const handledJob = useRef("");
  const generation = useAvatarGeneration();
  const upload = useMutation({
    mutationFn: (file: File) =>
      uploadSubmissionMediaDirect(file, "expert-avatar"),
  });
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
    generation.reset();
    setCategory(category);
    setShade("standard");
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
      setUploadError("Choose a PNG, JPEG, or WEBP under 5MB.");
      return;
    }
    setUploadError(null);
    try {
      const response = await upload.mutateAsync(file);
      if (typeof response !== "string" || !response.trim())
        throw new Error("No image URL");
      generation.reset();
      setSelectedUrl(response.trim());
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
    void generation.generate({
      category,
      shade,
      shape,
      base,
      tilt,
      inlay,
      accent_placement: accentPlacement,
      accent_count: accentCount,
      expression,
    });
  }
  function openFilePicker() {
    fileInputRef.current?.click();
  }

  return {
    selectedUrl,
    category,
    shape,
    setShape,
    shade,
    setShade,
    setCategory,
    base,
    setBase,
    tilt,
    setTilt,
    inlay,
    setInlay,
    accentPlacement,
    setAccentPlacement,
    accentCount,
    setAccentCount,
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
