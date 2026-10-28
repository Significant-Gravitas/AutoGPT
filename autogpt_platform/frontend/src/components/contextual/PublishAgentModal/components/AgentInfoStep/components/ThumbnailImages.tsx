"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import Image from "next/image";
import { isLocalStoreMediaUrl } from "@/lib/store-media";
import { useThumbnailImages } from "./useThumbnailImages";
import { cn } from "@/lib/utils";
import {
  Cancel01Icon,
  Loading03Icon,
  MagicWand01Icon,
  PlusSignIcon,
} from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";

interface ThumbnailImagesProps {
  agentId: string | null;
  onImagesChange: (images: string[]) => void;
  initialImages?: string[];
  initialSelectedImage?: string | null;
  errorMessage?: string;
}

export function ThumbnailImages({
  agentId,
  onImagesChange,
  initialImages = [],
  initialSelectedImage = null,
  errorMessage,
}: ThumbnailImagesProps) {
  const {
    images,
    selectedImage,
    isGenerating,
    isUploading,
    thumbnailsContainerRef,
    handleRemoveImage,
    handleAddImage,
    handleFileChange,
    handleGenerateImage,
    handleImageSelect,
  } = useThumbnailImages({
    agentId,
    onImagesChange,
    initialImages,
    initialSelectedImage,
  });

  return (
    <div className="space-y-4">
      <div className="flex flex-col gap-1">
        <div className="flex flex-col gap-1 sm:flex-row sm:items-center sm:justify-between">
          <Text variant="body-medium" as="h3" className="text-textBlack">
            Media
          </Text>
          <Text variant="small" as="span" className="text-foreground-tertiary">
            {images.length}/5 images
          </Text>
        </div>
        <Text variant="small" className="text-foreground-tertiary">
          The selected image is submitted first and appears as the marketplace
          thumbnail.
        </Text>
        {errorMessage && <p className="text-sm text-error">{errorMessage}</p>}
      </div>

      <div
        ref={thumbnailsContainerRef}
        className="relative z-10 flex flex-col gap-3"
      >
        {images.length === 0 ? (
          <div className="flex flex-col gap-2 sm:flex-row">
            <label
              htmlFor="image-upload"
              aria-disabled={isUploading}
              data-testid="thumbnail-add-image-empty"
              className={cn(
                // TODO(tokens): no exact role for border-zinc-700 #505057, hover:border-zinc-700 #505057; #505057 is foreground-body, a text colour
                "inline-flex h-[2.25rem] min-w-[5.5rem] cursor-pointer items-center justify-center gap-1.5 whitespace-nowrap rounded-full border border-zinc-700 bg-transparent px-3 py-2 font-sans text-sm font-medium leading-snug text-foreground-base transition-colors hover:border-zinc-700 hover:bg-surface-hover focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-focus",
                isUploading && "pointer-events-none opacity-60",
              )}
            >
              <input
                id="image-upload"
                type="file"
                accept="image/*"
                onChange={handleFileChange}
                disabled={isUploading}
                className="hidden"
              />
              {isUploading ? (
                <Icon icon={Loading03Icon} size={16} className="animate-spin" />
              ) : (
                <Icon icon={PlusSignIcon} size={16} />
              )}
              <span>{isUploading ? "Uploading" : "Add image"}</span>
            </label>
            <Button
              type="button"
              variant="outline"
              size="small"
              onClick={handleGenerateImage}
              disabled={isGenerating || isUploading || images.length >= 5}
              loading={isGenerating}
              leftIcon={<Icon icon={MagicWand01Icon} className="h-4 w-4" />}
            >
              {isGenerating ? "Generating" : "Generate"}
            </Button>
          </div>
        ) : (
          <div className="flex flex-col gap-3">
            <div className="flex gap-2 overflow-x-auto pb-1">
              {images.map((src, index) => (
                <div key={src} className="relative shrink-0">
                  <button
                    type="button"
                    onClick={() => handleImageSelect(src)}
                    className={cn(
                      // TODO(tokens): no exact role for focus-visible:ring-zinc-900 #2C2C30; #2C2C30 is action-hover, the button fill
                      "relative aspect-video h-28 w-44 overflow-hidden rounded-[10px] border-2 transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-zinc-900 focus-visible:ring-offset-2",
                      selectedImage === src
                        ? "border-action-hover"
                        : "border-transparent hover:border-border-strong",
                    )}
                    aria-pressed={selectedImage === src}
                    aria-label={`Select thumbnail ${index + 1}`}
                  >
                    <Image
                      src={src}
                      unoptimized={isLocalStoreMediaUrl(src)}
                      alt={`Thumbnail ${index + 1}`}
                      fill
                      style={{ objectFit: "cover" }}
                    />
                  </button>
                  <button
                    type="button"
                    onClick={(e) => {
                      e.preventDefault();
                      e.stopPropagation();
                      handleRemoveImage(index);
                    }}
                    className="absolute right-1.5 top-1.5 z-30 inline-flex size-7 cursor-pointer items-center justify-center rounded-full bg-action-hover/85 text-action-foreground shadow-sm backdrop-blur-sm transition-colors hover:bg-action-hover"
                    aria-label={`Remove image ${index + 1}`}
                    data-testid={`thumbnail-remove-${index}`}
                  >
                    <Icon icon={Cancel01Icon} size={14} />
                  </button>
                  {index === 0 ? (
                    <span className="mt-1 block text-center text-caption-sm font-medium leading-inherit text-foreground-tertiary">
                      Thumbnail
                    </span>
                  ) : null}
                </div>
              ))}
            </div>

            <div className="flex flex-col gap-2 sm:flex-row">
              {images.length < 5 ? (
                <Button
                  type="button"
                  onClick={handleAddImage}
                  variant="outline"
                  size="small"
                  disabled={isUploading || isGenerating}
                  loading={isUploading}
                  leftIcon={
                    isUploading ? undefined : (
                      <Icon icon={PlusSignIcon} size={16} />
                    )
                  }
                  data-testid="thumbnail-add-image"
                >
                  {isUploading ? "Uploading" : "Add image"}
                </Button>
              ) : null}
              <Button
                type="button"
                variant="outline"
                size="small"
                onClick={handleGenerateImage}
                disabled={isGenerating || isUploading || images.length >= 5}
                loading={isGenerating}
                leftIcon={<Icon icon={MagicWand01Icon} className="h-4 w-4" />}
              >
                {isGenerating
                  ? "Generating"
                  : images.length >= 5
                    ? "Max images reached"
                    : "Generate"}
              </Button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
