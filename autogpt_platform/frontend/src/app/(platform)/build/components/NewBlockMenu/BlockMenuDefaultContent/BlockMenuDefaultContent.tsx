import { Text } from "@/components/atoms/Text/Text";
import React from "react";

export const BlockMenuDefaultContent = () => {

  return (
    <div className="h-full flex-1 overflow-hidden flex items-center justify-center">
      {/* I have added temporary content here, will fillup it in follow up prs */}
      <Text variant="body" className="text-green-300"> 
        This is the block menu default content
      </Text>
    </div>
  );
};