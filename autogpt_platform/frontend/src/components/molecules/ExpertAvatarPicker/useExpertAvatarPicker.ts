import { ExpertAvatarRequestCategory } from "@/app/api/__generated__/models/expertAvatarRequestCategory";
import type { ExpertAvatarRequestBase } from "@/app/api/__generated__/models/expertAvatarRequestBase";
import type { ExpertAvatarRequestExpression } from "@/app/api/__generated__/models/expertAvatarRequestExpression";
import type { ExpertAvatarRequestInlay } from "@/app/api/__generated__/models/expertAvatarRequestInlay";
import type { ExpertAvatarRequestShape } from "@/app/api/__generated__/models/expertAvatarRequestShape";
import type { ExpertAvatarRequestTilt } from "@/app/api/__generated__/models/expertAvatarRequestTilt";
import { uploadSubmissionMediaDirect } from "@/lib/direct-upload";
import { useMutation } from "@tanstack/react-query";
import { useEffect, useRef, useState } from "react";
import {
  DEFAULT_EXPERT_AVATAR_URL,
  getExpertVisualCategory,
  getManagedIdentity,
  resolveExpertAvatarUrl,
} from "../ExpertAvatar/helpers";
import { ACCEPTED_AVATAR_TYPES, MAX_AVATAR_BYTES } from "./helpers";
import { useAvatarGeneration } from "./useAvatarGeneration";

interface Args {
  avatarUrl?: string | null;
  categories?: readonly string[] | null;
  color: string | null;
  onPick: (url: string, color: string) => void;
}

function toGenerationCategory(value: string): ExpertAvatarRequestCategory {
  return (
    Object.values(ExpertAvatarRequestCategory).find((c) => c === value) ??
    "general"
  );
}

export function useExpertAvatarPicker({
  avatarUrl,
  categories,
  color,
  onPick,
}: Args) {
  const [selectedUrl, setSelectedUrl] = useState(() =>
    resolveExpertAvatarUrl(avatarUrl),
  );
  // The saved identity, when this Expert has one, so the picker can offer it
  // back after a look at the alternatives.
  const savedIdentity = getManagedIdentity(avatarUrl);
  const [category, setCategory] = useState<ExpertAvatarRequestCategory>(() =>
    toGenerationCategory(getExpertVisualCategory(avatarUrl, categories)),
  );
  const [shape, setShape] = useState<ExpertAvatarRequestShape>("pebble");
  const [base, setBase] = useState<ExpertAvatarRequestBase>("compact");
  const [tilt, setTilt] = useState<ExpertAvatarRequestTilt>("level");
  const [inlay, setInlay] = useState<ExpertAvatarRequestInlay>("sweep");
  const [expression, setExpression] =
    useState<ExpertAvatarRequestExpression>("friendly");
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
  }, [generation.job]);

  const catalogUrls = [
    ...(savedIdentity && savedIdentity.url !== DEFAULT_EXPERT_AVATAR_URL
      ? [savedIdentity.url]
      : []),
    DEFAULT_EXPERT_AVATAR_URL,
  ];

  function selectCatalog(url: string) {
    if (!catalogUrls.includes(url)) return;
    generation.reset();
    setSelectedUrl(url);
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
        "Could not upload that picture. Try again or keep a managed look.",
      );
    }
  }

  function confirm() {
    onPick(selectedUrl, color ?? "amber-300");
  }
  function generate() {
    void generation.generate({
      category,
      shape,
      base,
      tilt,
      inlay,
      expression,
    });
  }
  function openFilePicker() {
    fileInputRef.current?.click();
  }

  return {
    selectedUrl,
    catalogUrls,
    category,
    setCategory,
    shape,
    setShape,
    base,
    setBase,
    tilt,
    setTilt,
    inlay,
    setInlay,
    expression,
    setExpression,
    selectCatalog,
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
