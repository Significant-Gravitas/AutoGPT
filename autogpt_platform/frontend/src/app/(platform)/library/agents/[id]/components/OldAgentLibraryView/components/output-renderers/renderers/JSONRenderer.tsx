"use client";

import React, { useState } from "react";
import { ChevronDown, ChevronRight } from "lucide-react";
import {
  OutputRenderer,
  OutputMetadata,
  DownloadContent,
  CopyContent,
} from "../types";

export class JSONRenderer implements OutputRenderer {
  name = "JSONRenderer";
  priority = 20;

  canRender(value: any, metadata?: OutputMetadata): boolean {
    if (metadata?.type === "json") {
      return true;
    }

    if (typeof value === "object" && value !== null) {
      return true;
    }

    if (typeof value === "string") {
      try {
        JSON.parse(value);
        return true;
      } catch {
        return false;
      }
    }

    return false;
  }

  render(value: any, metadata?: OutputMetadata): React.ReactNode {
    let jsonData = value;

    if (typeof value === "string") {
      try {
        jsonData = JSON.parse(value);
      } catch {
        return null;
      }
    }

    return <JSONViewer data={jsonData} />;
  }

  getCopyContent(value: any, metadata?: OutputMetadata): CopyContent | null {
    const jsonString =
      typeof value === "string" ? value : JSON.stringify(value, null, 2);

    return {
      mimeType: "application/json",
      data: jsonString,
      alternativeMimeTypes: ["text/plain"],
      fallbackText: jsonString,
    };
  }

  getDownloadContent(
    value: any,
    metadata?: OutputMetadata,
  ): DownloadContent | null {
    const jsonString =
      typeof value === "string" ? value : JSON.stringify(value, null, 2);
    const blob = new Blob([jsonString], { type: "application/json" });

    return {
      data: blob,
      filename: metadata?.filename || "output.json",
      mimeType: "application/json",
    };
  }

  isConcatenable(value: any, metadata?: OutputMetadata): boolean {
    return true;
  }
}

function JSONViewer({ data, depth = 0 }: { data: any; depth?: number }) {
  const [collapsed, setCollapsed] = useState<Record<string, boolean>>({});

  const toggleCollapse = (key: string) => {
    setCollapsed((prev) => ({ ...prev, [key]: !prev[key] }));
  };

  const renderValue = (value: any, key: string = ""): React.ReactNode => {
    if (value === null)
      return <span className="text-muted-foreground">null</span>;
    if (value === undefined)
      return <span className="text-muted-foreground">undefined</span>;

    if (typeof value === "boolean") {
      return <span className="text-blue-600">{value.toString()}</span>;
    }

    if (typeof value === "number") {
      return <span className="text-green-600">{value}</span>;
    }

    if (typeof value === "string") {
      return <span className="text-orange-600">"{value}"</span>;
    }

    if (Array.isArray(value)) {
      const isCollapsed = collapsed[key];
      const itemCount = value.length;

      if (itemCount === 0) {
        return <span className="text-muted-foreground">[]</span>;
      }

      return (
        <div className="inline-block">
          <button
            onClick={() => toggleCollapse(key)}
            className="inline-flex items-center rounded px-1 hover:bg-muted"
          >
            {isCollapsed ? (
              <ChevronRight className="h-3 w-3" />
            ) : (
              <ChevronDown className="h-3 w-3" />
            )}
            <span className="ml-1 text-muted-foreground">
              Array({itemCount})
            </span>
          </button>
          {!isCollapsed && (
            <div className="ml-4 mt-1">
              {value.map((item, index) => (
                <div key={index} className="flex">
                  <span className="mr-2 text-muted-foreground">{index}:</span>
                  {renderValue(item, `${key}[${index}]`)}
                </div>
              ))}
            </div>
          )}
        </div>
      );
    }

    if (typeof value === "object") {
      const isCollapsed = collapsed[key];
      const keys = Object.keys(value);

      if (keys.length === 0) {
        return <span className="text-muted-foreground">{"{}"}</span>;
      }

      return (
        <div className="inline-block">
          <button
            onClick={() => toggleCollapse(key)}
            className="inline-flex items-center rounded px-1 hover:bg-muted"
          >
            {isCollapsed ? (
              <ChevronRight className="h-3 w-3" />
            ) : (
              <ChevronDown className="h-3 w-3" />
            )}
            <span className="ml-1 text-muted-foreground">Object</span>
          </button>
          {!isCollapsed && (
            <div className="ml-4 mt-1">
              {keys.map((objKey) => (
                <div key={objKey} className="flex">
                  <span className="mr-2 text-purple-600">"{objKey}":</span>
                  {renderValue(value[objKey], `${key}.${objKey}`)}
                </div>
              ))}
            </div>
          )}
        </div>
      );
    }

    return <span className="text-muted-foreground">{String(value)}</span>;
  };

  return (
    <div className="overflow-x-auto rounded-md bg-muted p-3 font-mono text-sm">
      {renderValue(data, "root")}
    </div>
  );
}
