import { useRef, useState, type ComponentRef } from "react";
import { useWindowDimensions, type View } from "react-native";

const GAP = 6;
const EDGE = 16;
const MIN_WIDTH = 200;
const MAX_HEIGHT = 320;

interface Anchor {
  x: number;
  y: number;
  height: number;
}

export function useRecipientMenu() {
  const pillRef = useRef<ComponentRef<typeof View>>(null);
  const [anchor, setAnchor] = useState<Anchor | null>(null);
  const window = useWindowDimensions();

  function open() {
    pillRef.current?.measureInWindow((x, y, _width, height) => {
      setAnchor({ x, y, height });
    });
  }

  function close() {
    setAnchor(null);
  }

  // Just under the pill, left-aligned with it, kept on screen.
  const placement = anchor
    ? {
        top: anchor.y + anchor.height + GAP,
        left: Math.max(EDGE, Math.min(anchor.x, window.width - MIN_WIDTH - EDGE)),
        maxHeight: Math.min(
          MAX_HEIGHT,
          window.height - (anchor.y + anchor.height + GAP) - EDGE,
        ),
      }
    : null;

  return { pillRef, placement, isOpen: anchor !== null, open, close };
}
