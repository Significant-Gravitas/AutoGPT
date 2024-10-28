import * as React from "react";
import { X } from "lucide-react";
import Image from "next/image";
import { Button } from "../agptui/Button";

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
      className="inline-flex min-h-screen sm:h-auto sm:min-h-[824px] w-full sm:max-w-[670px] flex-col rounded-none sm:rounded-3xl border border-slate-200 bg-white shadow"
      role="dialog"
      aria-labelledby="modal-title"
    >
      <div className="w-full relative h-[180px] sm:h-[140px] rounded-none sm:rounded-t-3xl border-b border-slate-200">
        <div className="w-full absolute left-0 top-[40px] sm:top-[40px] flex flex-col items-center justify-start px-6">
          <div 
            id="modal-title"
            className="text-neutral-900 text-xl sm:text-2xl font-semibold font-['Poppins'] leading-relaxed mb-4 sm:mb-2 text-center"
          >
            Agent is awaiting review
          </div>
          <div className="text-center text-slate-500 text-sm font-normal font-['Inter'] leading-relaxed max-w-[280px] sm:max-w-none">
            In the meantime you can check your progress on your Creator Dashboard page
          </div>
        </div>
        <button 
          onClick={onClose}
          className="absolute right-4 top-4 h-[38px] w-[38px]"
          aria-label="Close dialog"
        >
          <X className="h-6 w-6 text-slate-500" />
        </button>
      </div>

      <div className="flex flex-1 flex-col items-center px-6 py-6 gap-8 sm:gap-6">
        <div className="flex w-full flex-col items-center gap-6 sm:gap-4 mt-4 sm:mt-0">
          <div className="flex flex-col items-center gap-3 sm:gap-2">
            <div className="font-['Geist'] text-lg font-semibold leading-7 text-neutral-800 text-center">
              {agentName}
            </div>
            <div className="font-['Geist'] text-base font-normal leading-normal text-neutral-600 text-center max-w-[280px] sm:max-w-none">
              {subheader}
            </div>
          </div>

          <div 
            className="w-full h-[280px] sm:h-[350px] bg-neutral-200 rounded-xl"
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
            className="w-full h-[150px] sm:h-[180px] overflow-y-auto font-['Geist'] text-base font-normal leading-normal text-neutral-600"
            tabIndex={0}
            role="region"
            aria-label="Agent description"
          >
            {description}
          </div>
        </div>
      </div>

      <div className="w-full p-6 flex flex-col sm:flex-row items-center justify-center gap-4 border-t border-slate-200">
        <Button
          onClick={onDone}
          variant="outline"
          className="w-full sm:flex-1 h-12 rounded-[59px]"
        >
          Done
        </Button>
        <Button
          onClick={onViewProgress}
          variant="default"
          className="w-full sm:flex-1 h-12 rounded-[59px] bg-neutral-800 hover:bg-neutral-900 text-white"
        >
          View progress
        </Button>
      </div>
    </div>
  );
};
