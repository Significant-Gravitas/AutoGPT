"use client";

import * as React from "react";
import Image from "next/image";
import { IconCross, IconPlus } from "../../../../../ui/icons";
import { Button } from "@/components/atoms/Button/Button";
import { MagicWand } from "@phosphor-icons/react";
import { useThumbnailImages } from "./useThumbnailImages";

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
    <div className="space-y-2.5">
      <div className="flex items-center justify-start gap-2">
        <label className="text-sm font-medium leading-tight text-slate-950 dark:text-slate-300">
          Thumbnail images
        </label>
        {errorMessage && <p className="text-sm text-red-500">{errorMessage}</p>}
      </div>
      <div className="flex h-[300px] items-center justify-center overflow-hidden rounded-md border border-neutral-300 p-2.5 dark:border-neutral-600">
        {selectedImage !== null && selectedImage !== undefined ? (
          <Image
            src={selectedImage}
            alt="Selected Thumbnail"
            width={500}
            height={350}
            style={{ objectFit: "cover" }}
            className="h-auto w-full rounded-md object-cover"
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
            <label
              htmlFor="image-upload"
              className="inline-flex h-[2.25rem] cursor-pointer items-center justify-center gap-1.5 whitespace-nowrap rounded-full border border-zinc-700 bg-transparent px-3 py-2 font-sans text-sm font-medium leading-snug text-black transition-colors hover:border-zinc-700 hover:bg-zinc-100 focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-neutral-950 disabled:pointer-events-none disabled:opacity-50"
            >
              <input
                id="image-upload"
                type="file"
                accept="image/*"
                onChange={handleFileChange}
                className="hidden"
              />
              <IconPlus className="h-4 w-4" />
              <span>Add image</span>
            </label>
            <Button
              type="button"
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
              <div
                key={index}
                className="relative flex-shrink-0 overflow-visible"
              >
                <Button
                  type="button"
                  size="small"
                  onClick={() => handleRemoveImage(index)}
                  className="absolute right-0 top-0 z-50 h-6 w-6 p-0"
                  aria-label="Remove image"
                >
                  <IconCross className="h-2 w-2 text-white" />
                </Button>
                <Image
                  src={src}
                  alt={`Thumbnail ${index + 1}`}
                  width={100}
                  height={70}
                  style={{ objectFit: "cover" }}
                  className="cursor-pointer rounded-md"
                  onClick={() => handleImageSelect(src)}
                />
              </div>
            ))}
            {images.length < 5 ? (
              <Button
                type="button"
                onClick={handleAddImage}
                variant="outline"
                size="small"
                className="!ml-4"
              >
                <IconPlus className="h-4 w-4" />
                <span>Add image</span>
              </Button>
            ) : null}
          </>
        )}
      </div>
    </div>
  );
}
