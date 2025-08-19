import { SmileySadIcon } from "@phosphor-icons/react";

export const NoSearchResult = () => {
  return (
    <div className="flex h-full w-full flex-col items-center justify-center text-center">
      <SmileySadIcon size={64} className="mb-10 text-zinc-400" />
      <div className="space-y-1">
        <p className="font-sans text-sm font-medium leading-[1.375rem] text-zinc-800">
          No match found
        </p>
        <p className="font-sans text-sm font-normal leading-[1.375rem] text-zinc-600">
          Try adjusting your search terms
        </p>
      </div>
    </div>
  );
};