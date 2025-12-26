import { useState, useRef, useEffect } from "react";
import BackendAPI from "@/lib/autogpt-server-api";
import { useToast } from "@/components/molecules/Toast/use-toast";

interface UseThumbnailImagesProps {
  agentId: string | null;
  onImagesChange: (images: string[]) => void;
  initialImages?: string[];
  initialSelectedImage?: string | null;
}

export function useThumbnailImages({
  agentId,
  onImagesChange,
  initialImages = [],
  initialSelectedImage = null,
}: UseThumbnailImagesProps) {
  const [images, setImages] = useState<string[]>(initialImages);

  const [selectedImage, setSelectedImage] = useState<string | null>(
    initialSelectedImage,
  );

  const [isGenerating, setIsGenerating] = useState(false);
  const thumbnailsContainerRef = useRef<HTMLDivElement | null>(null);
  const { toast } = useToast();

  // Memoize the stringified version to detect actual changes
  const initialImagesKey = JSON.stringify(initialImages);

  // Update images when initialImages prop changes (by value, not reference)
  useEffect(() => {
    if (initialImages.length > 0) {
      setImages(initialImages);
      setSelectedImage(initialSelectedImage || initialImages[0]);
    }
  }, [initialImagesKey, initialSelectedImage]); // Use stringified key instead of array reference

  // Notify parent when images change
  useEffect(() => {
    onImagesChange(images);
  }, [images, onImagesChange]);

  const setImagesWithValidation = (newImages: string[]) => {
    // Remove duplicates
    const uniqueImages = Array.from(new Set(newImages));
    // Keep only first 5 images
    const limitedImages = uniqueImages.slice(0, 5);
    setImages(limitedImages);
  };

  function handleRemoveImage(indexToRemove: number) {
    const newImages = [...images];
    newImages.splice(indexToRemove, 1);
    setImagesWithValidation(newImages);
    if (newImages[indexToRemove] === selectedImage) {
      setSelectedImage(newImages[0] || null);
    }
    if (newImages.length === 0) {
      setSelectedImage(null);
    }
  }

  async function handleAddImage() {
    if (images.length >= 5) return;

    const input = document.createElement("input");
    input.type = "file";
    input.accept = "image/*";

    // Create a promise that resolves when file is selected
    const fileSelected = new Promise<File | null>((resolve) => {
      input.onchange = (e) => {
        const file = (e.target as HTMLInputElement).files?.[0];
        resolve(file || null);
      };
    });

    // Trigger file selection
    input.click();

    // Wait for file selection
    const file = await fileSelected;
    if (!file) return;

    await uploadImage(file);
  }

  async function handleFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    if (images.length >= 5) return;

    const file = e.target.files?.[0];
    if (!file) return;

    await uploadImage(file);
  }

  async function uploadImage(file: File) {
    try {
      const api = new BackendAPI();

      const imageUrl = (await api.uploadStoreSubmissionMedia(file)).replace(
        /^"(.*)"$/,
        "$1",
      );

      setImagesWithValidation([...images, imageUrl]);
      if (!selectedImage) {
        setSelectedImage(imageUrl);
      }
    } catch (_error) {
      toast({
        title: "Upload failed",
        description: "Failed to upload image. Please try again.",
        variant: "destructive",
      });
    }
  }

  async function handleGenerateImage() {
    if (isGenerating || images.length >= 5) return;

    setIsGenerating(true);
    try {
      const api = new BackendAPI();
      if (!agentId) {
        throw new Error("Agent ID is required");
      }
      const { image_url } = await api.generateStoreSubmissionImage(agentId);
      setImagesWithValidation([...images, image_url]);
      if (!selectedImage) {
        setSelectedImage(image_url);
      }
    } catch (_error) {
      toast({
        title: "Generation failed",
        description: "Failed to generate image. Please try again.",
        variant: "destructive",
      });
    } finally {
      setIsGenerating(false);
    }
  }

  function handleImageSelect(imageSrc: string) {
    setSelectedImage(imageSrc);
  }

  return {
    // State
    images,
    selectedImage,
    isGenerating,
    thumbnailsContainerRef,
    // Handlers
    handleRemoveImage,
    handleAddImage,
    handleFileChange,
    handleGenerateImage,
    handleImageSelect,
  };
}
