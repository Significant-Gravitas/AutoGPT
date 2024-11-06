import * as React from "react";
import Image from "next/image";
import { Button } from "../agptui/Button";
import { IconClose, IconPlus } from "../ui/icons";

interface PublishAgentInfoProps {
  onBack: () => void;
  onSubmit: () => void;
  onClose: () => void;
  initialData?: {
    title: string;
    subheader: string;
    thumbnailSrc: string;
    youtubeLink: string;
    category: string;
    description: string;
    additionalImages?: string[];
  };
}

export const PublishAgentInfo: React.FC<PublishAgentInfoProps> = ({
  onBack,
  onSubmit,
  onClose,
  initialData,
}) => {
  const [images, setImages] = React.useState<string[]>(
    initialData?.additionalImages
      ? [initialData.thumbnailSrc, ...initialData.additionalImages]
      : initialData?.thumbnailSrc
        ? [initialData.thumbnailSrc]
        : [],
  );
  const [selectedImage, setSelectedImage] = React.useState<string | null>(
    initialData?.thumbnailSrc || null,
  );
  const thumbnailsContainerRef = React.useRef<HTMLDivElement | null>(null);

  const handleRemoveImage = (indexToRemove: number) => {
    console.log(`Remove image at index: ${indexToRemove}`);
    // Placeholder function for removing an image
  };

  const handleAddImage = () => {
    console.log("Add image button clicked");
    // Placeholder function for adding an image
  };

  return (
    <div className="mx-auto flex w-full max-w-[670px] flex-col rounded-3xl border border-slate-200 bg-white shadow-lg">
      <div className="relative border-b border-slate-200 p-6">
        <div className="absolute right-4 top-2">
          <button
            onClick={onClose}
            className="flex h-[38px] w-[38px] items-center justify-center rounded-full bg-gray-100 transition-colors hover:bg-gray-200"
            aria-label="Close"
          >
            <IconClose size="default" className="text-neutral-600" />
          </button>
        </div>
        <h2 className="text-center font-['Poppins'] text-2xl font-semibold leading-loose text-neutral-900">
          Publish Agent
        </h2>
        <p className="text-center font-['Geist'] text-base font-normal leading-7 text-neutral-600">
          Write a bit of details about your agent
        </p>
      </div>

      <div className="flex-grow space-y-5 overflow-y-auto p-6">
        <div className="space-y-1.5">
          <label
            htmlFor="title"
            className="font-['Geist'] text-sm font-medium leading-tight text-slate-950"
          >
            Title
          </label>
          <input
            id="title"
            type="text"
            placeholder="Agent name"
            defaultValue={initialData?.title}
            className="w-full rounded-[55px] border border-slate-200 py-2.5 pl-4 pr-14 font-['Geist'] text-base font-normal leading-normal text-slate-500"
          />
        </div>

        <div className="space-y-1.5">
          <label
            htmlFor="subheader"
            className="font-['Geist'] text-sm font-medium leading-tight text-slate-950"
          >
            Subheader
          </label>
          <input
            id="subheader"
            type="text"
            placeholder="A tagline for your agent"
            defaultValue={initialData?.subheader}
            className="w-full rounded-[55px] border border-slate-200 py-2.5 pl-4 pr-14 font-['Geist'] text-base font-normal leading-normal text-slate-500"
          />
        </div>

        <div className="space-y-2.5">
          <label className="font-['Geist'] text-sm font-medium leading-tight text-slate-950">
            Thumbnail images
          </label>
          <div className="flex h-[350px] items-center justify-center overflow-hidden rounded-[20px] border border-neutral-300 p-2.5">
            {selectedImage ? (
              <Image
                src={selectedImage}
                alt="Selected Thumbnail"
                width={500}
                height={350}
                objectFit="cover"
                className="rounded-md"
              />
            ) : (
              <p className="font-['Geist'] text-sm font-normal text-neutral-600">
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
                  className="flex h-[70px] w-[100px] flex-col items-center justify-center rounded-md bg-neutral-200 hover:bg-neutral-300"
                >
                  <IconPlus size="lg" className="text-neutral-600" />
                  <span className="mt-1 font-['Geist'] text-xs font-normal text-neutral-600">
                    Add image
                  </span>
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
                      objectFit="cover"
                      className="cursor-pointer rounded-md"
                      onClick={() => setSelectedImage(src)}
                    />
                    <button
                      onClick={() => handleRemoveImage(index)}
                      className="absolute right-1 top-1 flex h-5 w-5 items-center justify-center rounded-full bg-white bg-opacity-70 transition-opacity hover:bg-opacity-100"
                      aria-label="Remove image"
                    >
                      <IconClose size="sm" className="text-neutral-600" />
                    </button>
                  </div>
                ))}
                <Button
                  onClick={handleAddImage}
                  variant="ghost"
                  className="flex h-[70px] w-[100px] flex-col items-center justify-center rounded-md bg-neutral-200 hover:bg-neutral-300"
                >
                  <IconPlus size="lg" className="text-neutral-600" />
                  <span className="mt-1 font-['Geist'] text-xs font-normal text-neutral-600">
                    Add image
                  </span>
                </Button>
              </>
            )}
          </div>
        </div>

        <div className="space-y-1.5">
          <label className="font-['Geist'] text-sm font-medium leading-tight text-slate-950">
            AI image generator
          </label>
          <div className="flex items-center justify-between">
            <p className="font-['Geist'] text-base font-normal leading-normal text-slate-700">
              You can use AI to generate a cover image for you
            </p>
            <Button
              variant="default"
              size="sm"
              className="bg-neutral-800 text-white hover:bg-neutral-900"
            >
              Generate
            </Button>
          </div>
        </div>

        <div className="space-y-1.5">
          <label
            htmlFor="youtube"
            className="font-['Geist'] text-sm font-medium leading-tight text-slate-950"
          >
            YouTube video link
          </label>
          <input
            id="youtube"
            type="text"
            placeholder="Paste a video link here"
            defaultValue={initialData?.youtubeLink}
            className="w-full rounded-[55px] border border-slate-200 py-2.5 pl-4 pr-14 font-['Geist'] text-base font-normal leading-normal text-slate-500"
          />
        </div>

        <div className="space-y-1.5">
          <label
            htmlFor="category"
            className="font-['Geist'] text-sm font-medium leading-tight text-slate-950"
          >
            Category
          </label>
          <select
            id="category"
            defaultValue={initialData?.category}
            className="w-full appearance-none rounded-[55px] border border-slate-200 py-2.5 pl-4 pr-5 font-['Geist'] text-base font-normal leading-normal text-slate-500"
          >
            <option value="">Select a category for your agent</option>
            <option value="SEO">SEO</option>
            {/* Add more options here */}
          </select>
        </div>

        <div className="space-y-1.5">
          <label
            htmlFor="description"
            className="font-['Geist'] text-sm font-medium leading-tight text-slate-950"
          >
            Description
          </label>
          <textarea
            id="description"
            placeholder="Describe your agent and what it does"
            defaultValue={initialData?.description}
            className="h-[100px] w-full resize-none rounded-2xl border border-slate-200 bg-white py-2.5 pl-4 pr-14 font-['Geist'] text-base font-normal leading-normal text-slate-900"
          ></textarea>
        </div>
      </div>

      <div className="flex justify-between gap-4 border-t border-slate-200 p-6">
        <Button
          onClick={onBack}
          variant="outline"
          size="default"
          className="w-full sm:flex-1"
        >
          Back
        </Button>
        <Button
          onClick={onSubmit}
          variant="default"
          size="default"
          className="w-full bg-neutral-800 text-white hover:bg-neutral-900 sm:flex-1"
        >
          Submit for review
        </Button>
      </div>
    </div>
  );
};
