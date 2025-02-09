"use client";
import { LibrarySearchBar } from "@/components/agptui/LibrarySearchBar";
import LibraryNotificationDropdown from "../LibraryNotificationDropdown";
import { LibraryUploadAgent } from "../LibraryUploadAgent";
import { cn } from "@/lib/utils";
import LibraryAgentFilter from "../LibraryAgentFilter";
import { useLibraryPageContext } from "../providers/LibraryAgentProvider";

interface LibraryActionHeaderProps {}

/**
 * LibraryActionHeader component - Renders a header with search, notifications and filters
 */
const LibraryActionHeader: React.FC<LibraryActionHeaderProps> = ({}) => {
  const { agents } = useLibraryPageContext();

  return (
    <>
      <div className="items-start justify-between bg-neutral-50 pb-4 md:flex">
        <div className={cn("relative flex-1 space-y-[32px]")}>
          <LibraryNotificationDropdown />

          <div className="flex items-center gap-[10px] p-2">
            <span className="w-[96px] font-poppin text-[18px] font-semibold leading-[28px] text-neutral-800">
              My agents
            </span>
            <span className="w-[70px] font-sans text-[14px] font-normal leading-6">
              {agents.length} agents
            </span>
          </div>
        </div>

        <LibrarySearchBar />
        <div className="flex flex-1 flex-col items-end space-y-[32px]">
          <LibraryUploadAgent />
          <div className="flex items-center gap-[10px] pl-2 pr-2 font-sans text-[14px] font-[500] leading-[24px] text-neutral-600">
            <LibraryAgentFilter />
          </div>
        </div>
      </div>

      {/* Mobile and tablet */}
      <div className="flex flex-col gap-4 bg-neutral-50 p-4 pt-[52px] md:hidden">
        <div className="flex w-full justify-between">
          <LibraryNotificationDropdown />
          <LibraryUploadAgent />
        </div>

        <div className="flex items-center justify-center">
          <LibrarySearchBar />
        </div>

        <div className="flex w-full justify-between">
          <div className="flex items-center gap-2">
            <span className="font-poppin text-[18px] font-semibold leading-[28px] text-neutral-800">
              My agents
            </span>
            <span className="font-sans text-[14px] font-normal leading-6">
              {agents.length} agents
            </span>
          </div>
          <LibraryAgentFilter />
        </div>
      </div>
    </>
  );
};

export default LibraryActionHeader;
