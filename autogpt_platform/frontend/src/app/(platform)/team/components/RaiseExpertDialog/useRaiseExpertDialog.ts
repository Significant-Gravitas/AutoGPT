import { getFileSizeError } from "@/lib/direct-upload";
import { startRaise } from "@/services/experts/raise-expert";
import { prefetchRaiseIntro } from "@/services/experts/raise-intro";
import { useQueryClient } from "@tanstack/react-query";
import { useRouter } from "next/navigation";
import { useEffect, useState, type ChangeEvent, type FormEvent } from "react";
import { v4 as uuidv4 } from "uuid";
import { ACCEPTED_IMAGE_TYPES, inferArea, MAX_IMAGE_MB } from "./helpers";
import { reportRaiseFailure } from "./reportRaiseFailure";

// Long enough to skip the keystrokes of a word, short enough that the opening
// is usually back before the user reaches the Create button.
const INTRO_PREFETCH_DELAY_MS = 350;

interface Args {
  initialJobTitle: string;
  onClose: () => void;
}

export function useRaiseExpertDialog({ initialJobTitle, onClose }: Args) {
  const queryClient = useQueryClient();
  const router = useRouter();
  const [name, setName] = useState("");
  const [jobTitle, setJobTitle] = useState(initialJobTitle);
  const [image, setImage] = useState<{ file: File; url: string } | null>(null);
  const [imageError, setImageError] = useState<string | null>(null);
  // Chosen when the dialog opens, so the expert's page can be prefetched
  // before Create is pressed and the navigation has nothing left to load.
  const [expertId] = useState(() => uuidv4({}));

  useEffect(() => {
    router.prefetch(`/team/${expertId}`);
  }, [router, expertId]);

  useEffect(() => {
    if (!name.trim()) return;
    const timer = setTimeout(
      () => prefetchRaiseIntro(queryClient, name, jobTitle),
      INTRO_PREFETCH_DELAY_MS,
    );
    return () => clearTimeout(timer);
  }, [queryClient, name, jobTitle]);

  function pickImage(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (!file) return;
    if (!ACCEPTED_IMAGE_TYPES.split(",").includes(file.type)) {
      setImageError("Use a PNG, JPEG or WebP image.");
      return;
    }
    const sizeError = getFileSizeError(file, MAX_IMAGE_MB);
    if (sizeError) {
      setImageError(sizeError);
      return;
    }
    if (image) URL.revokeObjectURL(image.url);
    setImageError(null);
    setImage({ file, url: URL.createObjectURL(file) });
  }

  function removeImage() {
    if (image) URL.revokeObjectURL(image.url);
    setImage(null);
    setImageError(null);
  }

  function submit(event: FormEvent) {
    event.preventDefault();
    const trimmedName = name.trim();
    if (!trimmedName) return;
    const trimmedTitle = jobTitle.trim();
    prefetchRaiseIntro(queryClient, trimmedName, trimmedTitle);
    startRaise(queryClient, {
      id: expertId,
      name: trimmedName,
      jobTitle: trimmedTitle,
      ...inferArea(trimmedName, trimmedTitle),
      imageFile: image?.file ?? null,
      imagePreviewUrl: image?.url ?? null,
    }).catch((error: unknown) => {
      reportRaiseFailure(error, trimmedName);
      router.replace("/team");
    });
    onClose();
    router.push(`/team/${expertId}?welcome=1`);
  }

  return {
    name,
    setName,
    jobTitle,
    setJobTitle,
    imageUrl: image?.url ?? null,
    imageError,
    canSubmit: name.trim().length > 0,
    pickImage,
    removeImage,
    submit,
  };
}
