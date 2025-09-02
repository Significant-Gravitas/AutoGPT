"use client";

import React, { useCallback, useMemo } from "react";
import { Flag, useGetFlag } from "@/services/feature-flags/use-get-flag";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

import LoadingBox from "@/components/ui/loading";
import { CopyIcon, DownloadIcon, ShareIcon } from "lucide-react";
import { globalRegistry, OutputItem, OutputActions } from "./output-renderers";
import type { OutputMetadata } from "./output-renderers";

export function AgentRunOutputView({
  agentRunOutputs,
}: {
  agentRunOutputs:
    | Record<
        string,
        {
          title?: string;
          /* type: BlockIOSubType; */
          values: Array<React.ReactNode>;
        }
      >
    | undefined;
}) {
  const enableEnhancedOutputHandling = useGetFlag(
    Flag.ENABLE_ENHANCED_OUTPUT_HANDLING,
  );

  // Prepare items for the renderer system
  const outputItems = useMemo(() => {
    if (!agentRunOutputs) return [];

    const items: Array<{
      key: string;
      label: string;
      value: any;
      metadata?: OutputMetadata;
      renderer: any;
    }> = [];

    Object.entries(agentRunOutputs).forEach(([key, { title, values }]) => {
      values.forEach((value, index) => {
        const metadata: OutputMetadata = {
          // You can add metadata extraction logic here based on value type
        };

        const renderer = globalRegistry.getRenderer(value, metadata);
        if (renderer) {
          items.push({
            key: `${key}-${index}`,
            label: index === 0 ? title || key : "",
            value,
            metadata,
            renderer,
          });
        }
      });
    });

    return items;
  }, [agentRunOutputs]);

  return (
    <>
      {enableEnhancedOutputHandling ? (
        <Card className="agpt-box">
          <CardHeader>
            <div className="flex items-center justify-between">
              <CardTitle className="font-poppins text-lg">Output</CardTitle>
              {outputItems.length > 0 && (
                <OutputActions
                  items={outputItems.map((item) => ({
                    value: item.value,
                    metadata: item.metadata,
                    renderer: item.renderer,
                  }))}
                />
              )}
            </div>
          </CardHeader>

          <CardContent className="flex flex-col gap-4">
            {agentRunOutputs !== undefined ? (
              outputItems.length > 0 ? (
                outputItems.map((item) => (
                  <OutputItem
                    key={item.key}
                    value={item.value}
                    metadata={item.metadata}
                    renderer={item.renderer}
                    label={item.label}
                  />
                ))
              ) : (
                <p className="text-sm text-muted-foreground">
                  No outputs to display
                </p>
              )
            ) : (
              <LoadingBox spinnerSize={12} className="h-24" />
            )}
          </CardContent>
        </Card>
      ) : (
        <Card className="agpt-box">
          <CardHeader>
            <CardTitle className="font-poppins text-lg">Output</CardTitle>
          </CardHeader>

          <CardContent className="flex flex-col gap-4">
            {agentRunOutputs !== undefined ? (
              Object.entries(agentRunOutputs).map(
                ([key, { title, values }]) => (
                  <div key={key} className="flex flex-col gap-1.5">
                    <label className="text-sm font-medium">
                      {title || key}
                    </label>
                    {values.map((value, i) => (
                      <p
                        className="resize-none overflow-x-auto whitespace-pre-wrap break-words border-none text-sm text-neutral-700 disabled:cursor-not-allowed"
                        key={i}
                      >
                        {value}
                      </p>
                    ))}
                    {/* TODO: pretty type-dependent rendering */}
                  </div>
                ),
              )
            ) : (
              <LoadingBox spinnerSize={12} className="h-24" />
            )}
          </CardContent>
        </Card>
      )}
    </>
  );
}
