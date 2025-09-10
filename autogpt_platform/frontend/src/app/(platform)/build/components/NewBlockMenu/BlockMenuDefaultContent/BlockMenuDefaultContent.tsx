import { Text } from "@/components/atoms/Text/Text";
import React from "react";

export const BlockMenuDefaultContent = () => {
  return (
    <div className="flex h-full flex-1 items-center justify-center overflow-hidden">
      {/* I have added temporary content here, will fillup it in follow up prs */}
      <Text variant="body" className="text-green-300">
        This is the block menu default content
      </Text>
    </div>
  );
};
