import { useNodeStore } from "@/app/(platform)/build/stores/nodeStore";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { beautifyString } from "@/lib/utils";
import {
  ArrowSquareInIcon,
  CaretDownIcon,
  CopyIcon,
  InfoIcon,
} from "@phosphor-icons/react";
import { useState } from "react";

import { useShallow } from "zustand/react/shallow";

export const NodeDataRenderer = ({ nodeId }: { nodeId: string }) => {
  const [isExpanded, setIsExpanded] = useState(true);

  const nodeExecutionResult = useNodeStore(
    useShallow((state) => state.getNodeExecutionResult(nodeId)),
  );

  const data = {
    "[Input]": nodeExecutionResult?.input_data,
    ...nodeExecutionResult?.output_data,
  };

  // Don't render if there's no data
  if (!nodeExecutionResult || Object.keys(data).length === 0) {
    return null;
  }

  //  Need to Fix - when we are on build page and try to rerun the graph again, it gives error

  return (
    <div className="flex flex-col gap-3 rounded-b-xl border-t border-slate-200/50 px-4 py-4">
      <div className="flex items-center justify-between">
        <Text variant="body-medium" className="!font-semibold text-slate-700">
          Node Output
        </Text>
        <Button
          variant="ghost"
          size="small"
          onClick={() => setIsExpanded(!isExpanded)}
          className="h-fit min-w-0 p-1 text-slate-600 hover:text-slate-900"
        >
          <CaretDownIcon
            size={16}
            weight="bold"
            className={`transition-transform ${isExpanded ? "rotate-180" : ""}`}
          />
        </Button>
      </div>

      {isExpanded && (
        <>
          <div className="flex max-w-[350px] flex-col gap-4">
            {Object.entries(data || {}).map(([key, value]) => (
              <div key={key} className="flex flex-col gap-2">
                <div className="flex items-center gap-2">
                  <Text
                    variant="body-medium"
                    className="!font-semibold text-slate-600"
                  >
                    Pin:
                  </Text>
                  <Text variant="body" className="text-slate-700">
                    {beautifyString(key)}
                  </Text>
                </div>
                <div className="w-full space-y-2">
                  <Text
                    variant="small"
                    className="!font-semibold text-slate-600"
                  >
                    Data:
                  </Text>
                  <div className="relative">
                    <Text
                      variant="small"
                      className="rounded-xlarge bg-zinc-50 p-3 text-slate-700"
                    >
                      {JSON.stringify(value, null, 2)}
                    </Text>
                    <div className="mt-1 flex justify-end gap-1">
                      {/* TODO: Add tooltip for each button and also make all these blocks working */}
                      <Button
                        variant="secondary"
                        size="small"
                        className="h-fit min-w-0 gap-1.5 p-2 text-black hover:text-slate-900"
                      >
                        <InfoIcon size={16} />
                      </Button>
                      <Button
                        variant="secondary"
                        size="small"
                        className="h-fit min-w-0 gap-1.5 p-2 text-black hover:text-slate-900"
                      >
                        <ArrowSquareInIcon size={16} />
                      </Button>
                      <Button
                        variant="secondary"
                        size="small"
                        className="h-fit min-w-0 gap-1.5 p-2 text-black hover:text-slate-900"
                      >
                        <CopyIcon size={16} />
                      </Button>
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>

          {/* TODO: Currently this button is not working, need to make it working */}
          <Button variant="outline" size="small" className="w-fit self-start">
            View More
          </Button>
        </>
      )}
    </div>
  );
};
