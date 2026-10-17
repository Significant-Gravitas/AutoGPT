import {
  autoUpdate,
  flip,
  offset,
  shift,
  size,
  useFloating,
  type Middleware,
} from "@floating-ui/react-dom";

const VIEWPORT_MARGIN = 8;
const ANCHOR_OFFSET = 8;
const LIST_MAX_HEIGHT_REM = 20;

// The height cap from the last placement must not decide the next one: results
// that replace the short loading row would be measured at the old cap and
// never flip to the side where they fit. Place on the content's full height,
// once per computation, so size()'s re-measure of the capped box is kept.
const measureFullHeight: Middleware = {
  name: "measureFullHeight",
  fn({ rects, elements, middlewareData }) {
    if (middlewareData.measureFullHeight) return {};
    const list = elements.floating;
    const style = getComputedStyle(list);
    const rootFontSize = parseFloat(
      getComputedStyle(list.ownerDocument.documentElement).fontSize,
    );
    const height = Math.min(
      list.scrollHeight +
        parseFloat(style.borderTopWidth) +
        parseFloat(style.borderBottomWidth),
      LIST_MAX_HEIGHT_REM * rootFontSize,
    );
    return {
      data: { height },
      reset: { rects: { ...rects, floating: { ...rects.floating, height } } },
    };
  },
};

export function useMentionDropdown() {
  const { refs, floatingStyles } = useFloating({
    placement: "top-start",
    strategy: "fixed",
    whileElementsMounted: autoUpdate,
    middleware: [
      measureFullHeight,
      offset(ANCHOR_OFFSET),
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
