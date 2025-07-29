"use client";

import * as React from "react";
import Image from "next/image";
import { IconCross, IconPlus } from "../../../ui/icons";
import BackendAPI from "@/lib/autogpt-server-api";
import { toast } from "../../../molecules/Toast/use-toast";
import { Button } from "@/components/atoms/Button/Button";
import { StepHeader } from "./StepHeader";
import { MagicWand } from "@phosphor-icons/react";
import { Input } from "@/components/atoms/Input/Input";
import { Select } from "@/components/atoms/Select/Select";

export interface PublishAgentInfoInitialData {
  agent_id: string;
  title: string;
  subheader: string;
  slug: string;
  thumbnailSrc: string;
  youtubeLink: string;
  category: string;
  description: string;
  additionalImages?: string[];
}

interface PublishAgentInfoProps {
  onBack: () => void;
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

export const PublishAgentInfo: React.FC<PublishAgentInfoProps> = ({
  onBack,
  onSubmit,
  initialData,
}) => {
  const [agentId, setAgentId] = React.useState<string | null>(null);
  const [images, setImages] = React.useState<string[]>([]);
  const [selectedImage, setSelectedImage] = React.useState<string | null>(
    initialData?.thumbnailSrc || null,
  );
  const [title, setTitle] = React.useState(initialData?.title || "");
  const [subheader, setSubheader] = React.useState(
    initialData?.subheader || "",
  );
  const [youtubeLink, setYoutubeLink] = React.useState(
    initialData?.youtubeLink || "",
  );
  const [category, setCategory] = React.useState(initialData?.category || "");
  const [description, setDescription] = React.useState(
    initialData?.description || "",
  );
  const [slug, setSlug] = React.useState(initialData?.slug || "");
  const thumbnailsContainerRef = React.useRef<HTMLDivElement | null>(null);

  React.useEffect(() => {
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
        variant: "destructive",
      });
    }
  };

  const [isGenerating, setIsGenerating] = React.useState(false);

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

  return (
    <div className="mx-auto flex w-full flex-col rounded-3xl">
      <StepHeader
        title="Publish Agent"
        description="Write a bit of details about your agent"
      />

      <div className="flex-grow overflow-y-auto p-6">
        <Input
          label="Title"
          id="title"
          type="text"
          placeholder="Agent name"
          value={title}
          onChange={(e) => setTitle(e.target.value)}
        />

        <Input
          label="Subheader"
          id="subheader"
          type="text"
          placeholder="A tagline for your agent"
          value={subheader}
          onChange={(e) => setSubheader(e.target.value)}
        />

        <Input
          label="Slug"
          id="slug"
          type="text"
          placeholder="URL-friendly name for your agent"
          value={slug}
          onChange={(e) => setSlug(e.target.value)}
        />

        <div className="space-y-2.5">
          <label className="text-sm font-medium leading-tight text-slate-950 dark:text-slate-300">
            Thumbnail images
          </label>
          <div className="flex h-[250px] items-center justify-center overflow-hidden rounded-[20px] border border-neutral-300 p-2.5 dark:border-neutral-600">
            {selectedImage !== null && selectedImage !== undefined ? (
              <Image
                src={selectedImage}
                alt="Selected Thumbnail"
                width={500}
                height={350}
                style={{ objectFit: "cover" }}
                className="rounded-md"
              />
            ) : (
              <p className="font-sans text-sm font-normal text-neutral-600 dark:text-neutral-400">
                No images yet
              </p>
            )}
          </div>
          <div
            ref={thumbnailsContainerRef}
            className="flex items-center space-x-2 overflow-x-auto pb-6"
          >
            {images.length === 0 ? (
              <div className="flex w-full items-center justify-start gap-2 pl-2">
                <Button onClick={handleAddImage} variant="outline" size="small">
                  <label
                    htmlFor="image-upload"
                    className="flex-start flex items-center gap-1"
                  >
                    <input
                      id="image-upload"
                      type="file"
                      accept="image/*"
                      onChange={handleAddImage}
                      className="hidden"
                    />
                    <IconPlus className="h-4 w-4" />
                    <span>Add image</span>
                  </label>
                </Button>
                <Button
                  variant="outline"
                  size="small"
                  onClick={handleGenerateImage}
                  disabled={isGenerating || images.length >= 5}
                >
                  <MagicWand className="h-4 w-4" />
                  {isGenerating
                    ? "Generating..."
                    : images.length >= 5
                      ? "Max images reached"
                      : "Generate"}
                </Button>
              </div>
            ) : (
              <>
                {images.map((src, index) => (
                  <div key={index} className="relative flex-shrink-0">
                    <Image
                      src={src}
                      alt={`Thumbnail ${index + 1}`}
                      width={100}
                      height={70}
                      style={{ objectFit: "cover" }}
                      className="cursor-pointer rounded-md"
                      onClick={() => setSelectedImage(src)}
                    />
                    <Button
                      variant="ghost"
                      onClick={() => handleRemoveImage(index)}
                      className="absolute right-1 top-1 flex h-5 w-5 items-center justify-center"
                      aria-label="Remove image"
                    >
                      <IconCross className="h-4 w-4" />
                    </Button>
                  </div>
                ))}
                {images.length < 5 ? (
                  <Button
                    onClick={handleAddImage}
                    variant="outline"
                    size="small"
                  >
                    <IconPlus className="h-4 w-4" />
                    <span>Add image</span>
                  </Button>
                ) : null}
              </>
            )}
          </div>
        </div>

        <Input
          label="YouTube video link"
          id="youtube"
          type="url"
          placeholder="Paste a video link here"
          value={youtubeLink}
          onChange={(e) => setYoutubeLink(e.target.value)}
        />

        <Select
          label="Category"
          id="category"
          placeholder="Select a category for your agent"
          value={category}
          onValueChange={(value) => setCategory(value)}
          options={[
            { value: "productivity", label: "Productivity" },
            { value: "writing", label: "Writing & Content" },
            { value: "development", label: "Development" },
            { value: "data", label: "Data & Analytics" },
            { value: "marketing", label: "Marketing & SEO" },
            { value: "research", label: "Research & Learning" },
            { value: "creative", label: "Creative & Design" },
            { value: "business", label: "Business & Finance" },
            { value: "personal", label: "Personal Assistant" },
            { value: "other", label: "Other" },
          ]}
        />

        <Input
          label="Description"
          id="description"
          type="textarea"
          placeholder="Describe your agent and what it does"
          value={description}
          onChange={(e) => setDescription(e.target.value)}
        />
      </div>

      <div className="flex justify-between gap-4 p-6">
        <Button onClick={onBack} variant="secondary" className="w-full">
          Back
        </Button>
        <Button onClick={handleSubmit} className="w-full">
          Submit for review
        </Button>
      </div>
    </div>
  );
};
