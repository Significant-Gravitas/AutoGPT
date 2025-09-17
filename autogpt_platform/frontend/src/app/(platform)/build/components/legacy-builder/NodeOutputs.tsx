import React, { useState } from "react";
import { ContentRenderer } from "../../../../../components/ui/render";
import { beautifyString } from "@/lib/utils";
import { Maximize2 } from "lucide-react";
import { Button } from "../../../../../components/ui/button";
import * as Separator from "@radix-ui/react-separator";
import ExpandableOutputDialog from "./ExpandableOutputDialog";

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
  const [expandedDialog, setExpandedDialog] = useState<{
    isOpen: boolean;
    execId: string;
    pinName: string;
    data: any[];
  } | null>(null);

  const openExpandedView = (pinName: string, pinData: any[]) => {
    setExpandedDialog({
      isOpen: true,
      execId: title || "Node Output",
      pinName,
      data: pinData,
    });
  };

  const closeExpandedView = () => {
    setExpandedDialog(null);
  };
  return (
    <div className="m-4 space-y-4">
      {title && <strong className="mt-2flex">{title}</strong>}
      {Object.entries(data).map(([pin, dataArray]) => (
        <div key={pin} className="group">
          <div className="flex items-center justify-between">
            <div className="flex items-center">
              <strong className="mr-2">Pin:</strong>
              <span>{beautifyString(pin)}</span>
            </div>
            {(truncateLongData || dataArray.length > 10) && (
              <Button
                variant="outline"
                size="sm"
                onClick={() => openExpandedView(pin, dataArray)}
                className="hidden items-center gap-1 group-hover:flex"
                title="Expand Full View"
              >
                <Maximize2 size={14} />
                Expand
              </Button>
            )}
          </div>
          <div className="mt-2">
            <strong className="mr-2">Data:</strong>
            <div className="mt-1">
              {dataArray.slice(0, 10).map((item, index) => (
                <React.Fragment key={index}>
                  <ContentRenderer
                    value={item}
                    truncateLongData={truncateLongData}
                  />
                  {index < Math.min(dataArray.length, 10) - 1 && ", "}
                </React.Fragment>
              ))}
              {dataArray.length > 10 && (
                <span style={{ color: "#888" }}>
                  <br />
                  <b>⋮</b>
                  <br />
                  <span>and {dataArray.length - 10} more</span>
                </span>
              )}
            </div>
            <Separator.Root className="my-4 h-[1px] bg-gray-300" />
          </div>
        </div>
      ))}

      {expandedDialog && (
        <ExpandableOutputDialog
          isOpen={expandedDialog.isOpen}
          onClose={closeExpandedView}
          execId={expandedDialog.execId}
          pinName={expandedDialog.pinName}
          data={expandedDialog.data}
        />
      )}
    </div>
  );
}
