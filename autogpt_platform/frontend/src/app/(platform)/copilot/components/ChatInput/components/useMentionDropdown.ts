import {
  autoUpdate,
  flip,
  offset,
  shift,
  size,
  useFloating,
} from "@floating-ui/react-dom";

const VIEWPORT_MARGIN = 8;

export function useMentionDropdown() {
  const { refs, floatingStyles } = useFloating({
    placement: "top-start",
    strategy: "fixed",
    whileElementsMounted: autoUpdate,
    middleware: [
      offset(8),
      flip({ padding: VIEWPORT_MARGIN }),
      shift({ padding: VIEWPORT_MARGIN }),
      size({
        padding: VIEWPORT_MARGIN,
        apply({ availableHeight, rects, elements }) {
          elements.floating.style.setProperty(
            "--mention-available-height",
            `${Math.max(0, availableHeight)}px`,
          );
          elements.floating.style.setProperty(
            "--mention-anchor-width",
            `${rects.reference.width}px`,
          );
        },
      }),
    ],
  });

  return {
    setAnchor: refs.setReference,
    setList: refs.setFloating,
    listStyles: floatingStyles,
  };
}
