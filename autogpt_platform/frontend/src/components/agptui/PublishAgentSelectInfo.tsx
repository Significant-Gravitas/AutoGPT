"use client";

import * as React from "react";
import Image from "next/image";
import { Button } from "../agptui/Button";
import { IconClose, IconPlus } from "../ui/icons";
import BackendAPI from "@/lib/autogpt-server-api";
import { toast } from "../ui/use-toast";
import { X } from "lucide-react";
import { Input } from "../ui/input";
import { Label } from "../ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "../ui/select";
import { Textarea } from "../ui/textarea";

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
  onClose: () => void;
  initialData?: PublishAgentInfoInitialData;
}

export const PublishAgentInfo: React.FC<PublishAgentInfoProps> = ({
  onBack,
  onSubmit,
  onClose,
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
    onSubmit(title, subheader, slug, description, images, youtubeLink, [
      category,
    ]);
  };

  return (
    <div className="mx-auto flex h-fit w-full max-w-2xl flex-col rounded-3xl bg-white">
      {/* Top section */}
      <div className="relative flex h-28 items-center justify-center border-b border-slate-200 dark:border-slate-700">
        {/* Cancel Button */}
        <div className="absolute right-4 top-4">
          <Button
            onClick={onClose}
            className="flex h-8 w-8 items-center justify-center rounded-full bg-transparent p-0 transition-colors hover:bg-gray-200"
            aria-label="Close"
          >
            <X className="h-4 w-4" />
          </Button>
        </div>
        {/* Content */}
        <div className="text-center">
          <h3 className="font-poppins text-2xl font-semibold text-neutral-900">
            Publish Agent
          </h3>
          <p className="font-sans text-base font-normal text-neutral-600">
            Write a bit of details about your agent{" "}
          </p>
        </div>
      </div>

      {/* Form fields */}
      <div className="h-[50vh] flex-grow space-y-5 overflow-y-auto p-4 md:h-[38rem] md:p-6">
        <div className="space-y-1.5">
          <Label
            htmlFor="title"
            className="font-sans text-sm font-medium text-[#020617]"
          >
            Title
          </Label>
          <Input
            id="title"
            type="text"
            placeholder="Agent name"
            value={title}
            onChange={(e) => setTitle(e.target.value)}
            className="h-11 rounded-full border border-[#E2E8F0] px-4 py-2.5 font-sans text-sm text-neutral-500 md:text-base"
          />
        </div>

        <div className="space-y-1.5">
          <Label
            htmlFor="subheader"
            className="font-sans text-sm font-medium text-[#020617]"
          >
            Subheader
          </Label>
          <Input
            id="subheader"
            type="text"
            placeholder="A tagline for your agent"
            value={subheader}
            onChange={(e) => setSubheader(e.target.value)}
            className="h-11 rounded-full border border-[#E2E8F0] px-4 py-2.5 font-sans text-sm text-neutral-500 md:text-base"
          />
        </div>

        <div className="space-y-1.5">
          <Label
            htmlFor="slug"
            className="font-sans text-sm font-medium text-[#020617]"
          >
            Slug
          </Label>
          <Input
            id="slug"
            type="text"
            placeholder="URL-friendly name for your agent"
            value={slug}
            onChange={(e) => setSlug(e.target.value)}
            className="h-11 rounded-full border border-[#E2E8F0] px-4 py-2.5 font-sans text-sm text-neutral-500 md:text-base"
          />
        </div>

        <div className="space-y-2.5">
          <Label className="font-sans text-sm font-medium text-[#020617]">
            Thumbnail images
          </Label>
          <div className="flex h-[350px] items-center justify-center overflow-hidden rounded-[20px] border border-dashed border-neutral-300 p-2.5 dark:border-neutral-600">
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
            className="flex items-center space-x-2 overflow-x-auto"
          >
            {images.length === 0 ? (
              <div className="flex w-full justify-center">
                <Button
                  onClick={handleAddImage}
                  variant="ghost"
                  className="h-[70px] w-[100px] flex-col items-center justify-center rounded-md bg-neutral-200 hover:bg-neutral-300 dark:bg-neutral-700 dark:hover:bg-neutral-600"
                >
                  <Label
                    htmlFor="image-upload"
                    className="flex flex-col items-center justify-center font-sans text-sm font-medium text-[#020617]"
                  >
                    <Input
                      id="image-upload"
                      type="file"
                      accept="image/*"
                      onChange={handleAddImage}
                      className="hidden"
                    />
                    <IconPlus
                      size="lg"
                      className="text-neutral-600 dark:text-neutral-300"
                    />
                    <span className="mt-1 font-sans text-sm font-normal text-neutral-600 dark:text-neutral-300">
                      Add image
                    </span>
                  </Label>
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
                    <button
                      onClick={() => handleRemoveImage(index)}
                      className="absolute right-1 top-1 flex h-5 w-5 items-center justify-center rounded-full bg-white bg-opacity-70 transition-opacity hover:bg-opacity-100 dark:bg-gray-800 dark:bg-opacity-70 dark:hover:bg-opacity-100"
                      aria-label="Remove image"
                    >
                      <IconClose
                        size="sm"
                        className="text-neutral-600 dark:text-neutral-300"
                      />
                    </button>
                  </div>
                ))}
                {images.length < 5 && (
                  <Button
                    onClick={handleAddImage}
                    variant="ghost"
                    className="flex h-[70px] w-[100px] flex-col items-center justify-center rounded-md bg-neutral-200 hover:bg-neutral-300 dark:bg-neutral-700 dark:hover:bg-neutral-600"
                  >
                    <IconPlus
                      size="lg"
                      className="text-neutral-600 dark:text-neutral-300"
                    />
                    <span className="mt-1 font-sans text-xs font-normal text-neutral-600 dark:text-neutral-300">
                      Add image
                    </span>
                  </Button>
                )}
              </>
            )}
          </div>
        </div>

        <div className="space-y-1.5">
          <Label className="font-sans text-sm font-medium text-[#020617]">
            AI image generator
          </Label>
          <div className="flex flex-col justify-between gap-2 md:flex-row md:items-center">
            <p className="font-sans text-sm text-neutral-500 md:text-base">
              You can use AI to generate a cover image for you
            </p>
            <Button
              className={`w-fit bg-neutral-800 font-sans text-white hover:bg-neutral-900 dark:bg-neutral-600 dark:hover:bg-neutral-500 ${
                images.length >= 5 ? "cursor-not-allowed opacity-50" : ""
              }`}
              onClick={handleGenerateImage}
              disabled={isGenerating || images.length >= 5}
            >
              {isGenerating
                ? "Generating..."
                : images.length >= 5
                  ? "Max images reached"
                  : "Generate"}
            </Button>
          </div>
        </div>

        <div className="space-y-1.5">
          <Label
            htmlFor="youtube"
            className="font-sans text-sm font-medium text-[#020617]"
          >
            YouTube video link
          </Label>
          <Input
            id="youtube"
            type="text"
            placeholder="Paste a video link here"
            value={youtubeLink}
            onChange={(e) => setYoutubeLink(e.target.value)}
            className="h-11 rounded-full border border-[#E2E8F0] px-4 py-2.5 font-sans text-sm text-neutral-500 md:text-base"
          />
        </div>

        <div className="space-y-1.5">
          <Label
            htmlFor="category"
            className="font-sans text-sm font-medium text-[#020617]"
          >
            Category
          </Label>
          <Select value={category} onValueChange={setCategory}>
            <SelectTrigger className="h-11 rounded-full border border-[#E2E8F0] px-4 py-2.5 font-sans text-sm text-neutral-500 md:text-base">
              <SelectValue placeholder="Select a category for your agent" />
            </SelectTrigger>
            <SelectContent className="font-sans">
              <SelectItem value="productivity">Productivity</SelectItem>
              <SelectItem value="writing">Writing & Content</SelectItem>
              <SelectItem value="development">Development</SelectItem>
              <SelectItem value="data">Data & Analytics</SelectItem>
              <SelectItem value="marketing">Marketing & SEO</SelectItem>
              <SelectItem value="research">Research & Learning</SelectItem>
              <SelectItem value="creative">Creative & Design</SelectItem>
              <SelectItem value="business">Business & Finance</SelectItem>
              <SelectItem value="personal">Personal Assistant</SelectItem>
              <SelectItem value="other">Other</SelectItem>
            </SelectContent>
          </Select>
        </div>

        <div className="space-y-1.5">
          <Label
            htmlFor="description"
            className="font-sans text-sm font-medium text-[#020617]"
          >
            Description
          </Label>
          <Textarea
            id="description"
            placeholder="Describe your agent and what it does"
            value={description}
            onChange={(e) => setDescription(e.target.value)}
            className="h-[100px] w-full resize-none rounded-2xl border border-[#E2E8F0] px-4 py-2.5 font-sans text-sm text-neutral-500 md:text-base"
          ></Textarea>
        </div>
      </div>

      {/* Bottom buttons */}
      <div className="flex justify-between gap-4 border-t border-slate-200 p-6 dark:border-slate-700">
        <Button
          onClick={onBack}
          size="lg"
          className="flex w-full items-center justify-center text-sm dark:border-slate-700 dark:text-slate-300 sm:flex-1 md:text-base"
        >
          Back
        </Button>
        <Button
          onClick={handleSubmit}
          size="lg"
          className="flex w-full items-center justify-center bg-neutral-800 text-sm text-white hover:bg-neutral-900 dark:bg-neutral-600 dark:hover:bg-neutral-500 sm:flex-1 md:text-base"
        >
          Submit for review
        </Button>
      </div>
    </div>
  );
};
