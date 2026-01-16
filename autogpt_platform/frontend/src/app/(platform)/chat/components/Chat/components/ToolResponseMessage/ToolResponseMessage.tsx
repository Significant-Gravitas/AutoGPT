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

    if (responseType === "agent_output") {
      const execution = parsedResult.execution as
        | {
            outputs?: Record<string, unknown[]>;
          }
        | null
        | undefined;
      const outputs = execution?.outputs || {};
      const message = parsedResult.message as string | undefined;

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
          {message && (
            <div className="rounded border p-4">
              <Text variant="small" className="text-neutral-600">
                {message}
              </Text>
            </div>
          )}
          {Object.keys(outputs).length > 0 && (
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
          )}
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

    // Handle other response types with a message field (e.g., understanding_updated)
    if (parsedResult.message && typeof parsedResult.message === "string") {
      // Format tool name from snake_case to Title Case
      const formattedToolName = toolName
        .split("_")
        .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
        .join(" ");

      // Clean up message - remove incomplete user_name references
      let cleanedMessage = parsedResult.message;
      // Remove "Updated understanding with: user_name" pattern if user_name is just a placeholder
      cleanedMessage = cleanedMessage.replace(
        /Updated understanding with:\s*user_name\.?\s*/gi,
        "",
      );
      // Remove standalone user_name references
      cleanedMessage = cleanedMessage.replace(/\buser_name\b\.?\s*/gi, "");
      cleanedMessage = cleanedMessage.trim();

      // Only show message if it has content after cleaning
      if (!cleanedMessage) {
        return (
          <div
            className={cn(
              "flex items-center justify-center gap-2 px-4 py-2",
              className,
            )}
          >
            <WrenchIcon
              size={14}
              weight="bold"
              className="flex-shrink-0 text-neutral-500"
            />
            <Text variant="small" className="text-neutral-500">
              {formattedToolName}
            </Text>
          </div>
        );
      }

      return (
        <div className={cn("space-y-2 px-4 py-2", className)}>
          <div className="flex items-center justify-center gap-2">
            <WrenchIcon
              size={14}
              weight="bold"
              className="flex-shrink-0 text-neutral-500"
            />
            <Text variant="small" className="text-neutral-500">
              {formattedToolName}
            </Text>
          </div>
          <div className="rounded border p-4">
            <Text variant="small" className="text-neutral-600">
              {cleanedMessage}
            </Text>
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
