import { usePostV2UploadSubmissionMedia } from "@/app/api/__generated__/endpoints/store/store";
import { ApiError } from "@/lib/autogpt-server-api/helpers";
import { toast } from "@/components/molecules/Toast/use-toast";
import {
  clampPart,
  colorForToken,
  hashSeed,
  notionAvatarUrlFor,
  notionConfigForName,
  randomNotionConfig,
  seededRandom,
  type NotionAvatarConfig,
} from "@/components/molecules/NotionAvatar/helpers";
import type { NotionCategory } from "@/components/molecules/NotionAvatar/metadata.generated";
import { useRef, useState } from "react";
import { COLOR_OPTIONS } from "../../../ColorStep/helpers";
import { ACCEPTED_AVATAR_TYPES, MAX_AVATAR_BYTES } from "../../helpers";

// The accept attribute only filters the OS picker, so the same list has to
// gate the upload for files that arrive by drag-and-drop or a widened filter.
const ACCEPTED_AVATAR_TYPE_LIST = ACCEPTED_AVATAR_TYPES.split(",");

interface Args {
  name: string;
  color: string | null;
  onPick: (avatarUrl: string, colorId: string) => void;
}

export function useNotionAvatarPicker({ name, color, onPick }: Args) {
  const [colorId, setColorId] = useState(() => color ?? seedColorToken(name));
  const colorIdRef = useRef(colorId);
  colorIdRef.current = colorId;
  const [config, setConfig] = useState<NotionAvatarConfig>(() =>
    seedConfig(name),
  );
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [isUploading, setIsUploading] = useState(false);
  const { mutateAsync: uploadMedia } = usePostV2UploadSubmissionMedia();

  // Marks stay off when shuffling too: blush and freckles are something to opt
  // into from the row, not something a generated face should arrive wearing.
  function shuffle() {
    setConfig(withoutMarks(randomNotionConfig()));
  }

  function cycle(category: NotionCategory, step: number) {
    setConfig((current) => ({
      ...current,
      parts: {
        ...current.parts,
        [category]: clampPart(category, current.parts[category] + step),
      },
    }));
  }

  function confirm() {
    onPick(notionAvatarUrlFor(withColor(config, colorId)), colorId);
  }

  function openFilePicker() {
    fileInputRef.current?.click();
  }

  async function handleFileChange(file: File | undefined) {
    if (!file) return;
    if (!ACCEPTED_AVATAR_TYPE_LIST.includes(file.type)) {
      toast({
        title: "That file isn't an image",
        description: "Pick a PNG, JPEG, or WebP image.",
        variant: "destructive",
      });
      return;
    }
    if (file.size > MAX_AVATAR_BYTES) {
      toast({
        title: "That image is too large",
        description: "Pick something under 5MB.",
        variant: "destructive",
      });
      return;
    }

    setIsUploading(true);
    try {
      const response = await uploadMedia({
        data: { file },
        params: { purpose: "expert-avatar" },
      });
      onPick(response.data as string, colorIdRef.current);
    } catch (error) {
      toast({
        title: "Couldn't save that picture",
        description:
          error instanceof ApiError
            ? error.message
            : "Try another image, or keep the drawn face.",
        variant: "destructive",
      });
    } finally {
      setIsUploading(false);
    }
  }

  return {
    config: withColor(config, colorId),
    colorId,
    setColorId,
    shuffle,
    cycle,
    confirm,
    fileInputRef,
    isUploading,
    openFilePicker,
    handleFileChange,
  };
}

/** The disc follows whichever accent the expert is wearing, so the preview and
 *  the chat theme never disagree. */
function withColor(
  config: NotionAvatarConfig,
  colorId: string,
): NotionAvatarConfig {
  const mapped = colorForToken(colorId);
  return mapped ? { ...config, color: mapped } : config;
}

function seedConfig(name: string): NotionAvatarConfig {
  return withoutMarks(notionConfigForName(name));
}

// A name-seeded accent so the step opens on something rather than nothing; the
// swatches are there to change it.
function seedColorToken(name: string): string {
  const random = seededRandom(hashSeed(`${name.toLowerCase()}:color`));
  return COLOR_OPTIONS[Math.floor(random() * COLOR_OPTIONS.length)].id;
}

function withoutMarks(config: NotionAvatarConfig): NotionAvatarConfig {
  return { ...config, parts: { ...config.parts, details: 0 } };
}
