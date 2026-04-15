import * as React from "react";

const HEADER_OFFSET = 100;

export function calculateScrollPosition(
  elementRect: DOMRect,
  containerRect: DOMRect,
  currentScrollTop: number,
): number {
  const elementTopRelativeToContainer =
    elementRect.top - containerRect.top + currentScrollTop - HEADER_OFFSET;

  return Math.max(0, elementTopRelativeToContainer);
}

function hasDisplayName(
  type: unknown,
  displayName: string,
): type is { displayName: string } {
  return (
    typeof type === "object" &&
    type !== null &&
    "displayName" in type &&
    (type as { displayName: unknown }).displayName === displayName
  );
}

export function findListElement(
  children: React.ReactNode[],
): React.ReactElement | undefined {
  return children.find(
    (child) =>
      React.isValidElement(child) &&
      hasDisplayName(child.type, "ScrollableTabsList"),
  ) as React.ReactElement | undefined;
}

export function findContentElements(
  children: React.ReactNode[],
): React.ReactNode[] {
  return children.filter(
    (child) =>
      !(
        React.isValidElement(child) &&
        hasDisplayName(child.type, "ScrollableTabsList")
      ),
  );
}
