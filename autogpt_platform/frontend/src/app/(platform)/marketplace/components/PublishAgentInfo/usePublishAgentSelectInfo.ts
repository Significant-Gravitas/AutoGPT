import { toast } from "@/components/ui/use-toast";
import BackendAPI from "@/lib/autogpt-server-api";
import { useEffect, useRef, useState } from "react";
import { PublishAgentInfoInitialData } from "./PublishAgentSelectInfo";

interface usePublishAgentInfoProps {
  onSubmit: (
    name: string,
    subHeading: string,
    slug: string,
    description: string,
    imageUrls: string[],
    videoUrl: string,
    categories: string[],
  ) => void;
  initialData?: PublishAgentInfoInitialData;
}

// Need to combine some states in one state
export const usePublishAgentSelectInfo = ({
  onSubmit,
  initialData,
}: usePublishAgentInfoProps) => {
  const [agentId, setAgentId] = useState<string | null>(null);
  const [images, setImages] = useState<string[]>([]);
  const [selectedImage, setSelectedImage] = useState<string | null>(
    initialData?.thumbnailSrc || null,
  );
  const [title, setTitle] = useState(initialData?.title || "");
  const [subheader, setSubheader] = useState(initialData?.subheader || "");
  const [youtubeLink, setYoutubeLink] = useState(
    initialData?.youtubeLink || "",
  );
  const [category, setCategory] = useState(initialData?.category || "");
  const [description, setDescription] = useState(
    initialData?.description || "",
  );
  const [slug, setSlug] = useState(initialData?.slug || "");

  const thumbnailsContainerRef = useRef<HTMLDivElement | null>(null);
  const [isGenerating, setIsGenerating] = useState(false);

  useEffect(() => {
    if (initialData) {
      setAgentId(initialData.agent_id);
      setImagesWithValidation([
        ...(initialData?.thumbnailSrc ? [initialData.thumbnailSrc] : []),
        ...(initialData.additionalImages || []),
      ]);
      setSelectedImage(initialData.thumbnailSrc || null);
      setTitle(initialData.title);
      setSubheader(initialData.subheader);
      setYoutubeLink(initialData.youtubeLink);
      setCategory(initialData.category);
      setDescription(initialData.description);
      setSlug(initialData.slug);
    }
  }, [initialData]);

  const setImagesWithValidation = (newImages: string[]) => {
    // Remove duplicates
    const uniqueImages = Array.from(new Set(newImages));
    // Keep only first 5 images
    const limitedImages = uniqueImages.slice(0, 5);
    setImages(limitedImages);
  };

  const handleRemoveImage = (indexToRemove: number) => {
    const newImages = [...images];
    newImages.splice(indexToRemove, 1);
    setImagesWithValidation(newImages);
    if (newImages[indexToRemove] === selectedImage) {
      setSelectedImage(newImages[0] || null);
    }
    if (newImages.length === 0) {
      setSelectedImage(null);
    }
  };

  const handleAddImage = async () => {
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
    } catch (error) {
      toast({
        title: "Failed to upload image",
        description: `Error: ${error}`,
      });
    }
  };

  const handleGenerateImage = async () => {
    if (isGenerating || images.length >= 5) return;

    setIsGenerating(true);
    try {
      const api = new BackendAPI();
      if (!agentId) {
        throw new Error("Agent ID is required");
      }
      const { image_url } = await api.generateStoreSubmissionImage(agentId);
      setImagesWithValidation([...images, image_url]);
    } catch (error) {
      console.error("Failed to generate image:", error);
    } finally {
      setIsGenerating(false);
    }
  };

  const handleSubmit = (e: React.MouseEvent<HTMLButtonElement>) => {
    e.preventDefault();
    const categories = category ? [category] : [];
    onSubmit(
      title,
      subheader,
      slug,
      description,
      images,
      youtubeLink,
      categories,
    );
  };

  return {
    images,
    selectedImage,
    setSelectedImage,
    title,
    setTitle,
    subheader,
    setSubheader,
    youtubeLink,
    setYoutubeLink,
    category,
    setCategory,
    description,
    setDescription,
    slug,
    setSlug,
    isGenerating,
    handleGenerateImage,
    handleSubmit,
    handleAddImage,
    handleRemoveImage,
    thumbnailsContainerRef,
  };
};
