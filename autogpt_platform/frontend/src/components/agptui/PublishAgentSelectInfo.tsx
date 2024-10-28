import * as React from "react";
import Image from "next/image";
import { Button } from "../agptui/Button";

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
      : []
  );
  const [selectedImage, setSelectedImage] = React.useState<string | null>(initialData?.thumbnailSrc || null);
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
    <div className="w-full max-w-[670px] bg-white rounded-3xl shadow-lg border border-slate-200 flex flex-col mx-auto">
      <div className="p-6 border-b border-slate-200 relative">
        <div className="absolute top-2 right-4">
          <button
            onClick={onClose}
            className="w-[38px] h-[38px] rounded-full bg-gray-100 flex items-center justify-center hover:bg-gray-200 transition-colors"
            aria-label="Close"
          >
            <svg
              width="14"
              height="14"
              viewBox="0 0 14 14"
              fill="none"
              xmlns="http://www.w3.org/2000/svg"
            >
              <path
                d="M1 1L13 13M1 13L13 1"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </svg>
          </button>
        </div>
        <h2 className="text-neutral-900 text-2xl font-semibold font-['Poppins'] leading-loose text-center">Publish Agent</h2>
        <p className="text-neutral-600 text-base font-normal font-['Geist'] leading-7 text-center">Write a bit of details about your agent</p>
      </div>
      
      <div className="flex-grow p-6 space-y-5 overflow-y-auto">
        <div className="space-y-1.5">
          <label htmlFor="title" className="text-slate-950 text-sm font-medium font-['Geist'] leading-tight">Title</label>
          <input
            id="title"
            type="text"
            placeholder="Agent name"
            defaultValue={initialData?.title}
            className="w-full pl-4 pr-14 py-2.5 rounded-[55px] border border-slate-200 text-slate-500 text-base font-normal font-['Geist'] leading-normal"
          />
        </div>

        <div className="space-y-1.5">
          <label htmlFor="subheader" className="text-slate-950 text-sm font-medium font-['Geist'] leading-tight">Subheader</label>
          <input
            id="subheader"
            type="text"
            placeholder="A tagline for your agent"
            defaultValue={initialData?.subheader}
            className="w-full pl-4 pr-14 py-2.5 rounded-[55px] border border-slate-200 text-slate-500 text-base font-normal font-['Geist'] leading-normal"
          />
        </div>

        <div className="space-y-2.5">
          <label className="text-slate-950 text-sm font-medium font-['Geist'] leading-tight">Thumbnail images</label>
          <div className="h-[350px] p-2.5 rounded-[20px] border border-neutral-300 flex items-center justify-center overflow-hidden">
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
              <p className="text-neutral-600 text-sm font-normal font-['Geist']">No images yet</p>
            )}
          </div>
          <div ref={thumbnailsContainerRef} className="flex items-center space-x-2 overflow-x-auto">
            {images.length === 0 ? (
              <div className="w-full flex justify-center">
                <Button
                  onClick={handleAddImage}
                  variant="ghost"
                  className="w-[100px] h-[70px] bg-neutral-200 rounded-md flex flex-col items-center justify-center hover:bg-neutral-300"
                >
                  <svg width="24" height="24" viewBox="0 0 28 28" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path d="M14 5.83334V22.1667" stroke="#666666" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                    <path d="M5.83331 14H22.1666" stroke="#666666" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                  </svg>
                  <span className="text-neutral-600 text-xs font-normal font-['Geist'] mt-1">Add image</span>
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
                      className="rounded-md cursor-pointer"
                      onClick={() => setSelectedImage(src)}
                    />
                    <button
                      onClick={() => handleRemoveImage(index)}
                      className="absolute top-1 right-1 w-5 h-5 bg-white bg-opacity-70 rounded-full flex items-center justify-center hover:bg-opacity-100 transition-opacity"
                      aria-label="Remove image"
                    >
                      <svg width="10" height="10" viewBox="0 0 14 14" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <path d="M1 1L13 13M1 13L13 1" stroke="#666666" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                      </svg>
                    </button>
                  </div>
                ))}
                <Button
                  onClick={handleAddImage}
                  variant="ghost"
                  className="w-[100px] h-[70px] bg-neutral-200 rounded-md flex flex-col items-center justify-center hover:bg-neutral-300"
                >
                  <svg width="24" height="24" viewBox="0 0 28 28" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path d="M14 5.83334V22.1667" stroke="#666666" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                    <path d="M5.83331 14H22.1666" stroke="#666666" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                  </svg>
                  <span className="text-neutral-600 text-xs font-normal font-['Geist'] mt-1">Add image</span>
                </Button>
              </>
            )}
          </div>
        </div>

        <div className="space-y-1.5">
          <label className="text-slate-950 text-sm font-medium font-['Geist'] leading-tight">AI image generator</label>
          <div className="flex items-center justify-between">
            <p className="text-slate-700 text-base font-normal font-['Geist'] leading-normal">You can use AI to generate a cover image for you</p>
            <Button
              variant="default"
              size="sm"
              className="text-white bg-neutral-800 hover:bg-neutral-900"
            >
              Generate
            </Button>
          </div>
        </div>

        <div className="space-y-1.5">
          <label htmlFor="youtube" className="text-slate-950 text-sm font-medium font-['Geist'] leading-tight">YouTube video link</label>
          <input
            id="youtube"
            type="text"
            placeholder="Paste a video link here"
            defaultValue={initialData?.youtubeLink}
            className="w-full pl-4 pr-14 py-2.5 rounded-[55px] border border-slate-200 text-slate-500 text-base font-normal font-['Geist'] leading-normal"
          />
        </div>

        <div className="space-y-1.5">
          <label htmlFor="category" className="text-slate-950 text-sm font-medium font-['Geist'] leading-tight">Category</label>
          <select
            id="category"
            defaultValue={initialData?.category}
            className="w-full pl-4 pr-5 py-2.5 rounded-[55px] border border-slate-200 text-slate-500 text-base font-normal font-['Geist'] leading-normal appearance-none"
          >
            <option value="">Select a category for your agent</option>
            <option value="SEO">SEO</option>
            {/* Add more options here */}
          </select>
        </div>

        <div className="space-y-1.5">
          <label htmlFor="description" className="text-slate-950 text-sm font-medium font-['Geist'] leading-tight">Description</label>
          <textarea
            id="description"
            placeholder="Describe your agent and what it does"
            defaultValue={initialData?.description}
            className="w-full h-[100px] pl-4 pr-14 py-2.5 rounded-2xl border border-slate-200 text-slate-900 text-base font-normal font-['Geist'] leading-normal resize-none bg-white"
          ></textarea>
        </div>
      </div>
      
      <div className="p-6 border-t border-slate-200 flex justify-between gap-4">
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
          className="w-full sm:flex-1 text-white bg-neutral-800 hover:bg-neutral-900"
        >
          Submit for review
        </Button>
      </div>
    </div>
  );
};
