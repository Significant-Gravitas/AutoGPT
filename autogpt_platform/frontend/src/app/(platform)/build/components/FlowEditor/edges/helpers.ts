// ------------------- JS Beads helpers -------------------

export const getLengthOfPathInPixels = (path: string) => {
  const pathElement = document.createElementNS(
    "http://www.w3.org/2000/svg",
    "path",
  );
  pathElement.setAttribute("d", path);
  return pathElement.getTotalLength();
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
