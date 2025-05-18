import React from "react";
import Block from "../Block";

const OutputBlocksContent: React.FC = () => {
  return (
    <div className="scrollbar-thin scrollbar-thumb-rounded scrollbar-thumb-zinc-200 scrollbar-track-transparent h-full overflow-y-scroll pt-4">
      <div className="w-full space-y-3 px-4 pb-4">
        <Block title="Date Input" description="Input a date into your agent." />
        <Block
          title="Dropdown input"
          description="Give your users the ability to select from a dropdown menu"
        />
        <Block title="File upload" description="Upload a file to your agent" />
      </div>
    </div>
  );
};

export default OutputBlocksContent;
