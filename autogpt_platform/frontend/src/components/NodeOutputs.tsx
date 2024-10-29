import React from "react";
import { ContentRenderer } from "./ui/render";
import { beautifyString } from "@/lib/utils";
import * as Separator from "@radix-ui/react-separator";

type NodeOutputsProps = {
  title?: string;
  truncateLongData?: boolean;
  data: { [key: string]: Array<any> };
};

export default function NodeOutputs({
  title,
  truncateLongData,
  data,
}: NodeOutputsProps) {
  return (
    <div className="m-4 space-y-4">
      {title && <strong className="mt-2flex">{title}</strong>}
      {Object.entries(data).map(([pin, dataArray]) => (
        <div key={pin} className="">
          <div className="flex items-center">
            <strong className="mr-2">Pin:</strong>
            <span>{beautifyString(pin)}</span>
          </div>
          <div className="mt-2">
            <strong className="mr-2">Data:</strong>
            <div className="mt-1">
              {dataArray.map((item, index) => (
                <React.Fragment key={index}>
                  <ContentRenderer
                    value={item}
                    truncateLongData={truncateLongData}
                  />
                  {index < dataArray.length - 1 && ", "}
                </React.Fragment>
              ))}
            </div>
            <Separator.Root className="my-4 h-[1px] bg-gray-300" />
          </div>
        </div>
      ))}
    </div>
  );
}
