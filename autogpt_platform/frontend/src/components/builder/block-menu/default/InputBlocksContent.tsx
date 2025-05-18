import React from "react";
import Block from "../Block";

const InputBlocksContent: React.FC = () => {
  return (
    <div className="scrollbar-thin scrollbar-thumb-rounded scrollbar-thumb-zinc-200 scrollbar-track-transparent h-full overflow-y-scroll pt-4">
      <div className="w-full space-y-3 px-4 pb-4">
        <Block title="Date Input" description="Input a date into your agent." />
        <Block
          title="Dropdown input"
          description="Give your users the ability to select from a dropdown menu"
        />
        <Block title="File upload" description="Upload a file to your agent" />
        <Block
          title="Text input"
          description="Allow users to select multiple options using checkboxes"
        />
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
        <Block
          title="Add to list"
          description="Enables your agent to chat with users in natural language."
        />
        <Block
          title="Add to list"
          description="Enables your agent to chat with users in natural language."
        />
      </div>
    </div>
  );
};

export default InputBlocksContent;
