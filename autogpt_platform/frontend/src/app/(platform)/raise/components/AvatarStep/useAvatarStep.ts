import { usePostV2UploadSubmissionMedia } from "@/app/api/__generated__/endpoints/store/store";
import { toast } from "@/components/molecules/Toast/use-toast";
import { useRef, useState } from "react";
import { ACCEPTED_AVATAR_TYPES, MAX_AVATAR_BYTES } from "./helpers";

// The accept attribute only filters the OS picker, so the same list has to
// gate the upload for files that arrive by drag-and-drop or a widened filter.
const ACCEPTED_AVATAR_TYPE_LIST = ACCEPTED_AVATAR_TYPES.split(",");

interface Args {
  onPick: (avatarUrl: string) => void;
}

export function useAvatarStep({ onPick }: Args) {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [isUploading, setIsUploading] = useState(false);
  const { mutateAsync: uploadMedia } = usePostV2UploadSubmissionMedia();

  function openFilePicker() {
    fileInputRef.current?.click();
  }

  async function handleFileChange(file: File | undefined) {
    if (!file) return;
    if (!ACCEPTED_AVATAR_TYPE_LIST.includes(file.type)) {
      toast({
        title: "That file isn't an image",
        description: "Pick a PNG, JPEG, WEBP, or GIF.",
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
      const response = await uploadMedia({ data: { file } });
      onPick(response.data as string);
    } catch {
      toast({
        title: "Couldn't save that picture",
        description: "Try another image, or skip for now.",
        variant: "destructive",
      });
    } finally {
      setIsUploading(false);
    }
  }

  return { fileInputRef, isUploading, openFilePicker, handleFileChange };
}
