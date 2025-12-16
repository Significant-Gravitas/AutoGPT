import { Text } from "@/components/atoms/Text/Text";
import "@/components/contextual/OutputRenderers";
import {
  globalRegistry,
  OutputItem,
} from "@/components/contextual/OutputRenderers";
import { cn } from "@/lib/utils";
import type { ToolResult } from "@/types/chat";
import { WrenchIcon } from "@phosphor-icons/react";
import { getToolActionPhrase } from "../../helpers";

export interface ToolResponseMessageProps {
  toolName: string;
  result?: ToolResult;
  success?: boolean;
  className?: string;
}

export function ToolResponseMessage({
  toolName,
  result,
  success: _success = true,
  className,
}: ToolResponseMessageProps) {
  if (!result) {
    return (
      <div className={cn("flex items-center justify-center gap-2", className)}>
        <WrenchIcon
          size={14}
          weight="bold"
          className="flex-shrink-0 text-neutral-500"
        />
        <Text variant="small" className="text-neutral-500">
          {getToolActionPhrase(toolName)}...
        </Text>
      </div>
    );
  }

  let parsedResult: Record<string, unknown> | null = null;
  try {
    parsedResult =
      typeof result === "string"
        ? JSON.parse(result)
        : (result as Record<string, unknown>);
  } catch {
    parsedResult = null;
  }

  if (parsedResult && typeof parsedResult === "object") {
    const responseType = parsedResult.type as string | undefined;

    if (responseType === "agent_output" && parsedResult.execution) {
      const execution = parsedResult.execution as {
        outputs?: Record<string, unknown[]>;
      };
      const outputs = execution.outputs || {};

      return (
        <div className={cn("space-y-4 px-4 py-2", className)}>
          <div className="flex items-center gap-2">
            <WrenchIcon
              size={14}
              weight="bold"
              className="flex-shrink-0 text-neutral-500"
            />
            <Text variant="small" className="text-neutral-500">
              {getToolActionPhrase(toolName)}
            </Text>
          </div>
          <div className="space-y-4">
            {Object.entries(outputs).map(([outputName, values]) =>
              values.map((value, index) => {
                const renderer = globalRegistry.getRenderer(value);
                if (renderer) {
                  return (
                    <OutputItem
                      key={`${outputName}-${index}`}
                      value={value}
                      renderer={renderer}
                      label={outputName}
                    />
                  );
                }
                return (
                  <div
                    key={`${outputName}-${index}`}
                    className="rounded border p-4"
                  >
                    <Text variant="large-medium" className="mb-2 capitalize">
                      {outputName}
                    </Text>
                    <pre className="overflow-auto text-sm">
                      {JSON.stringify(value, null, 2)}
                    </pre>
                  </div>
                );
              }),
            )}
          </div>
        </div>
      );
    }

    if (responseType === "block_output" && parsedResult.outputs) {
      const outputs = parsedResult.outputs as Record<string, unknown[]>;

      return (
        <div className={cn("space-y-4 px-4 py-2", className)}>
          <div className="flex items-center gap-2">
            <WrenchIcon
              size={14}
              weight="bold"
              className="flex-shrink-0 text-neutral-500"
            />
            <Text variant="small" className="text-neutral-500">
              {getToolActionPhrase(toolName)}
            </Text>
          </div>
          <div className="space-y-4">
            {Object.entries(outputs).map(([outputName, values]) =>
              values.map((value, index) => {
                const renderer = globalRegistry.getRenderer(value);
                if (renderer) {
                  return (
                    <OutputItem
                      key={`${outputName}-${index}`}
                      value={value}
                      renderer={renderer}
                      label={outputName}
                    />
                  );
                }
                return (
                  <div
                    key={`${outputName}-${index}`}
                    className="rounded border p-4"
                  >
                    <Text variant="large-medium" className="mb-2 capitalize">
                      {outputName}
                    </Text>
                    <pre className="overflow-auto text-sm">
                      {JSON.stringify(value, null, 2)}
                    </pre>
                  </div>
                );
              }),
            )}
          </div>
        </div>
      );
    }
  }

  const renderer = globalRegistry.getRenderer(result);
  if (renderer) {
    return (
      <div className={cn("px-4 py-2", className)}>
        <div className="mb-2 flex items-center gap-2">
          <WrenchIcon
            size={14}
            weight="bold"
            className="flex-shrink-0 text-neutral-500"
          />
          <Text variant="small" className="text-neutral-500">
            {getToolActionPhrase(toolName)}
          </Text>
        </div>
        <OutputItem value={result} renderer={renderer} />
      </div>
    );
  }

  return (
    <div className={cn("flex items-center justify-center gap-2", className)}>
      <WrenchIcon
        size={14}
        weight="bold"
        className="flex-shrink-0 text-neutral-500"
      />
      <Text variant="small" className="text-neutral-500">
        {getToolActionPhrase(toolName)}...
      </Text>
    </div>
  );
}
