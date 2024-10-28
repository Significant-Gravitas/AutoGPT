import * as React from "react";
import { X } from "lucide-react";
import Image from "next/image";

interface PublishAgentAwaitingReviewProps {
  agentName: string;
  subheader: string;
  description: string;
  thumbnailSrc?: string;
  onClose: () => void;
  onDone: () => void;
  onViewProgress: () => void;
}

export const PublishAgentAwaitingReview: React.FC<PublishAgentAwaitingReviewProps> = ({
  agentName,
  subheader,
  description,
  thumbnailSrc,
  onClose,
  onDone,
  onViewProgress,
}) => {
  return (
    <div 
      className="inline-flex h-[824px] w-[670px] flex-col rounded-3xl border border-slate-200 bg-white shadow"
      role="dialog"
      aria-labelledby="modal-title"
    >
      <div className="w-full relative h-[111px] rounded-t-3xl border border-slate-200">
        <div className="w-full absolute left-0 top-[36px] flex flex-col items-center justify-start">
          <div 
            id="modal-title"
            className="text-neutral-900 text-2xl font-semibold font-['Poppins'] leading-loose"
          >
            Agent is awaiting review
          </div>
          <div className="h-7 text-center text-slate-500 text-sm font-normal font-['Inter'] leading-tight">
            In the meantime you can check your progress on your Creator Dashboard page
          </div>
        </div>
        <button 
          onClick={onClose}
          className="absolute right-[8px] top-[8px] h-[38px] w-[38px]"
          aria-label="Close dialog"
        >
          <X className="h-6 w-6 text-slate-500" />
        </button>
      </div>

      <div className="flex flex-1 flex-col items-center px-6 pt-3 gap-5">
        <div className="flex w-full flex-col items-center gap-3">
          <div className="flex flex-col items-center gap-0.5">
            <div className="font-['Geist'] text-lg font-semibold leading-7 text-neutral-800">
              {agentName}
            </div>
            <div className="font-['Geist'] text-base font-normal leading-normal text-neutral-600">
              {subheader}
            </div>
          </div>
          
          <div 
            className="w-full h-[350px] bg-neutral-200 rounded-xl"
            role="img"
            aria-label={thumbnailSrc ? "Agent thumbnail" : "Thumbnail placeholder"}
          >
            {thumbnailSrc && (
              <Image
                src={thumbnailSrc}
                alt="Agent thumbnail"
                width={500}
                height={350}
                className="h-full w-full rounded-xl object-cover"
              />
            )}
          </div>

          <div 
            className="w-full h-[180px] overflow-y-auto font-['Geist'] text-base font-normal leading-normal text-neutral-600"
            tabIndex={0}
            role="region"
            aria-label="Agent description"
          >
            {description}
          </div>
        </div>
      </div>

      <div className="w-full p-6 flex items-center justify-center gap-4 rounded-b-3xl border border-slate-200">
        <button
          onClick={onDone}
          className="flex-1 h-10 flex items-center justify-center rounded-[59px] border border-neutral-900 bg-white"
        >
          <span className="font-['Geist'] text-sm font-medium leading-normal text-neutral-800">
            Done
          </span>
        </button>
        <button
          onClick={onViewProgress}
          className="flex-1 h-10 flex items-center justify-center rounded-[59px] bg-neutral-800"
        >
          <span className="font-['Geist'] text-sm font-medium leading-normal text-white">
            View progress
          </span>
        </button>
      </div>
    </div>
  );
};
