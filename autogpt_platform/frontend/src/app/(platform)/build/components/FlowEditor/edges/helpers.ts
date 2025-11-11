import { Link } from "@/app/api/__generated__/models/link";
import { Connection } from "@xyflow/react";

export const convertConnectionsToBackendLinks = (
  connections: Connection[],
): Link[] =>
  connections.map((c) => ({
    source_id: c.source || "",
    sink_id: c.target || "",
    source_name: c.sourceHandle || "",
    sink_name: c.targetHandle || "",
  }));

export const getLengthOfPathInPixels = (path: string) => {
  const pathElement = document.createElementNS(
    "http://www.w3.org/2000/svg",
    "path",
  );
  pathElement.setAttribute("d", path);
  return pathElement.getTotalLength();
};
