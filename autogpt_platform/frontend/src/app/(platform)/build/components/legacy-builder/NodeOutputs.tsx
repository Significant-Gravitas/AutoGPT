import React, { useContext, useMemo, useState } from "react";
import { Button } from "@/components/__legacy__/ui/button";
import { Maximize2 } from "lucide-react";
import * as Separator from "@radix-ui/react-separator";
import { ContentRenderer } from "@/components/__legacy__/ui/render";
import type { OutputMetadata } from "@/components/contextual/OutputRenderers";
import {
  globalRegistry,
  OutputItem,
} from "@/components/contextual/OutputRenderers";
import { Flag, useGetFlag } from "@/services/feature-flags/use-get-flag";

import { beautifyString } from "@/lib/utils";

import { BuilderContext } from "./Flow/Flow";
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
  const builderContext = useContext(BuilderContext);
  const enableEnhancedOutputHandling = useGetFlag(
    Flag.ENABLE_ENHANCED_OUTPUT_HANDLING,
  );

  const [expandedDialog, setExpandedDialog] = useState<{
    isOpen: boolean;
    execId: string;
    pinName: string;
    data: any[];
  } | null>(null);

  if (!builderContext) {
    throw new Error(
      "BuilderContext consumer must be inside FlowEditor component",
    );
  }

  const { getNodeTitle } = builderContext;

  // Prepare renderers for each item when enhanced mode is enabled
  const getItemRenderer = useMemo(() => {
    if (!enableEnhancedOutputHandling) return null;
    return (item: unknown) => {
      const metadata: OutputMetadata = {};
      return globalRegistry.getRenderer(item, metadata);
    };
  }, [enableEnhancedOutputHandling]);

  const getBeautifiedPinName = (pin: string) => {
    if (!pin.startsWith("tools_^_")) {
      return beautifyString(pin);
    }
    // Special handling for tool pins: replace node ID with node title
    const toolNodeID = pin.slice(8).split("_~_")[0]; // tools_^_{node_id}_~_{field}
    const toolNodeTitle = getNodeTitle(toolNodeID);
    return toolNodeTitle
      ? beautifyString(pin.replace(toolNodeID, toolNodeTitle))
      : beautifyString(pin);
  };

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
              <span>{getBeautifiedPinName(pin)}</span>
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
              {dataArray.slice(0, 10).map((item, index) => {
                const renderer = getItemRenderer?.(item);
                if (enableEnhancedOutputHandling && renderer) {
                  const metadata: OutputMetadata = {};
                  return (
                    <React.Fragment key={index}>
                      <OutputItem
                        value={item}
                        metadata={metadata}
                        renderer={renderer}
                      />
                      {index < Math.min(dataArray.length, 10) - 1 && ", "}
                    </React.Fragment>
                  );
                }
                return (
                  <React.Fragment key={index}>
                    <ContentRenderer
                      value={item}
                      truncateLongData={truncateLongData}
                    />
                    {index < Math.min(dataArray.length, 10) - 1 && ", "}
                  </React.Fragment>
                );
              })}
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
