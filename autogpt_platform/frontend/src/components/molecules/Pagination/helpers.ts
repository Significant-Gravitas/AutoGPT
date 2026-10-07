export type PageItem = number | "ellipsis-start" | "ellipsis-end";

function range(start: number, end: number): number[] {
  return Array.from({ length: end - start + 1 }, (_, index) => start + index);
}

/**
 * Page numbers to show: always the first and last page, `siblingCount` pages
 * either side of the current one, and an ellipsis for each gap of two or more.
 * The item count stays constant (siblingCount * 2 + 5) once pages overflow.
 */
export function getPageItems(
  page: number,
  pageCount: number,
  siblingCount = 1,
): PageItem[] {
  const totalSlots = siblingCount * 2 + 5;
  if (pageCount <= totalSlots) return range(1, pageCount);

  const current = Math.min(Math.max(page, 1), pageCount);
  const leftSibling = Math.max(current - siblingCount, 1);
  const rightSibling = Math.min(current + siblingCount, pageCount);
  const showStartEllipsis = leftSibling > 3;
  const showEndEllipsis = rightSibling < pageCount - 2;
  const edgeCount = 3 + siblingCount * 2;

  if (!showStartEllipsis) {
    return [...range(1, edgeCount), "ellipsis-end", pageCount];
  }
  if (!showEndEllipsis) {
    return [
      1,
      "ellipsis-start",
      ...range(pageCount - edgeCount + 1, pageCount),
    ];
  }
  return [
    1,
    "ellipsis-start",
    ...range(leftSibling, rightSibling),
    "ellipsis-end",
    pageCount,
  ];
}
