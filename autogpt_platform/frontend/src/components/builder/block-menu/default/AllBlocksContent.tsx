// BLOCK MENU TODO: Currently I have hide the scrollbar, but need to add better designed custom scroller

import React from "react";
import Block from "../Block";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";

const AllBlocksContent: React.FC = () => {
  return (
    <div className="scrollbar-thin scrollbar-thumb-rounded scrollbar-thumb-zinc-200 scrollbar-track-transparent h-full overflow-y-scroll pt-4">
      <div className="w-full space-y-3 px-4 pb-4">
        {/* AI Category */}
        <div className="space-y-2.5">
          <div className="flex items-center justify-between">
            <p className="font-sans text-sm font-medium leading-[1.375rem] text-zinc-800">
              AI
            </p>
            <span className="rounded-full bg-zinc-100 px-[0.375rem] font-sans text-sm leading-[1.375rem] text-zinc-600">
              10
            </span>
          </div>

          <div className="space-y-2">
            <Block
              title="Add to list"
              description="Enables your agent to chat with users in natural language."
            />
            <Block
              title="Add to list"
              description="Enables your agent to chat with users in natural language."
            />
            <Block
              title="Add to list"
              description="Enables your agent to chat with users in natural language."
            />

            <Button
              variant={"link"}
              className="px-0 font-sans text-sm leading-[1.375rem] text-zinc-600 underline hover:text-zinc-800"
            >
              see all
            </Button>
          </div>
        </div>

        <Separator className="h-[1px] w-full text-zinc-300" />

        {/* Basic Category */}
        <div className="space-y-2.5">
          <div className="flex items-center justify-between">
            <p className="font-sans text-sm font-medium leading-[1.375rem] text-zinc-800">
              Basic
            </p>
            <span className="rounded-full bg-zinc-100 px-[0.375rem] font-sans text-sm leading-[1.375rem] text-zinc-600">
              6
            </span>
          </div>

          <div className="space-y-2">
            <Block
              title="Add to list"
              description="Enables your agent to chat with users in natural language."
            />
            <Block
              title="Add to list"
              description="Enables your agent to chat with users in natural language."
            />
            <Block
              title="Add to list"
              description="Enables your agent to chat with users in natural language."
            />

            <Button
              variant={"link"}
              className="px-0 font-sans text-sm leading-[1.375rem] text-zinc-600 underline hover:text-zinc-800"
            >
              see all
            </Button>
          </div>
        </div>

        <Separator className="h-[1px] w-full text-zinc-300" />

        {/* Communincation Category */}
        <div className="space-y-2.5">
          <div className="flex items-center justify-between">
            <p className="font-sans text-sm font-medium leading-[1.375rem] text-zinc-800">
              Communincation
            </p>
            <span className="rounded-full bg-zinc-100 px-[0.375rem] font-sans text-sm leading-[1.375rem] text-zinc-600">
              6
            </span>
          </div>

          <div className="space-y-2">
            <Block
              title="Add to list"
              description="Enables your agent to chat with users in natural language."
            />
            <Block
              title="Add to list"
              description="Enables your agent to chat with users in natural language."
            />
            <Block
              title="Add to list"
              description="Enables your agent to chat with users in natural language."
            />

            <Button
              variant={"link"}
              className="px-0 font-sans text-sm leading-[1.375rem] text-zinc-600 underline hover:text-zinc-800"
            >
              see all
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AllBlocksContent;
