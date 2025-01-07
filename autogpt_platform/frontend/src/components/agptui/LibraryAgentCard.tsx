import { cn } from "@/lib/utils";
import Link from "next/link";
import { GraphMeta } from "@/lib/autogpt-server-api";

export const LibraryAgentCard = ({ id, name, isCreatedByUser }: GraphMeta) => {
  return (
    <div
      className={cn(
        "flex h-[158px] flex-col rounded-[14px] border border-[#E5E5E5] bg-white p-5 transition-all duration-300 ease-in-out hover:scale-[1.02]",
        !isCreatedByUser && "shadow-[0_-5px_0_0_rgb(196_181_253)]",
      )}
    >
      <div className="flex flex-1">
        <h3 className="flex-1 font-inter text-[18px] font-semibold leading-4">
          {name}
        </h3>
        {/* <span
          className={cn(
            "h-[14px] w-[14px] rounded-full",
            status == "Nothing running" && "bg-[#64748B]",
            status == "healthy" && "bg-[#22C55E]",
            status == "something wrong" && "bg-[#EF4444]",
            status == "waiting for trigger" && "bg-[#FBBF24]",
          )}
        ></span> */}
      </div>

      <div className="flex items-center justify-between">
        <div className="mt-6 flex gap-3">
          <Link
            href={`/agents/${id}`}
            className="font-inter text-[14px] font-[700] leading-[24px] text-neutral-800 hover:cursor-pointer hover:underline"
          >
            See runs
          </Link>

          {isCreatedByUser && (
            <Link
              href={`/build?flowID=${id}`}
              className="font-inter text-[14px] font-[700] leading-[24px] text-neutral-800 hover:underline"
            >
              Open in builder
            </Link>
          )}
        </div>
        {/* {output && (
          <div className="h-[24px] w-fit rounded-[45px] bg-neutral-600 px-[9px] py-[2px] font-sans text-[12px] font-[700] text-neutral-50">
            New output
          </div>
        )} */}
      </div>
    </div>
  );
};
