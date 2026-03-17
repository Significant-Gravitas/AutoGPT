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

// ------------------- SVG Beads helpers -------------------

export const getLengthOfPathInPixels = (path: string) => {
  const pathElement = document.createElementNS(
    "http://www.w3.org/2000/svg",
    "path",
  );
  pathElement.setAttribute("d", path);
  return pathElement.getTotalLength();
};

// ------------------- JS Beads helpers -------------------

export const getPointAtT = (
  t: number,
  edgePath: string,
  pathRef: React.MutableRefObject<SVGPathElement | null>,
) => {
  if (!pathRef.current) {
    const tempPath = document.createElementNS(
      "http://www.w3.org/2000/svg",
      "path",
    );
    tempPath.setAttribute("d", edgePath);
    pathRef.current = tempPath;
  }

  const totalLength = pathRef.current.getTotalLength();
  const point = pathRef.current.getPointAtLength(t * totalLength);
  return { x: point.x, y: point.y };
};

export const getTForDistance = (
  distanceFromEnd: number,
  totalLength: number,
) => {
  return Math.max(0, Math.min(1, 1 - distanceFromEnd / totalLength));
};

export const setTargetPositions = (
  beads: { t: number; targetT: number; startTime: number }[],
  beadDiameter: number,
  getTForDistanceFunc: (distanceFromEnd: number) => number,
) => {
  return beads.map((bead, index) => ({
    ...bead,
    targetT: getTForDistanceFunc(beadDiameter * (index + 1)),
  }));
};
