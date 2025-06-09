import { Frown } from "lucide-react";

const NoSearchResult = () => {
  return (
    <div className="flex h-full w-full flex-col items-center justify-center text-center">
      <Frown className="mb-10 h-16 w-16 text-zinc-400" strokeWidth={1} />
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

export default NoSearchResult;
