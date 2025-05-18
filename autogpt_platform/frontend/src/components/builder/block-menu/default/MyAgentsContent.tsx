import React from "react";
import UGCAgentBlock from "../UGCAgentBlock";

const MyAgentsContent: React.FC = () => {
  return (
    <div className="scrollbar-thin scrollbar-thumb-rounded scrollbar-thumb-zinc-200 scrollbar-track-transparent h-full overflow-y-scroll pt-4">
      <div className="w-full space-y-3 px-4 pb-4">
        <UGCAgentBlock
          title="My Agent 1"
          edited_time="23rd April"
          version={3}
          image_url="/placeholder.png"
        />
        <UGCAgentBlock
          title="My Agent 2"
          edited_time="21st April"
          version={4}
          image_url="/placeholder.png"
        />
        <UGCAgentBlock
          title="My Agent 3"
          edited_time="23rd May"
          version={7}
          image_url="/placeholder.png"
        />
        <UGCAgentBlock
          title="My Agent 4"
          edited_time="23rd April"
          version={3}
          image_url="/placeholder.png"
        />
        <UGCAgentBlock
          title="My Agent 5"
          edited_time="23rd April"
          version={3}
          image_url="/placeholder.png"
        />
        <UGCAgentBlock
          title="My Agent 6"
          edited_time="23rd April"
          version={3}
          image_url="/placeholder.png"
        />
      </div>
    </div>
  );
};

export default MyAgentsContent;
