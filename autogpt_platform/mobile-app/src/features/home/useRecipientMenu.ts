import { useRef, useState, type ComponentRef } from "react";
import { useWindowDimensions, type View } from "react-native";

const GAP = 6;
const EDGE = 16;
const MENU_WIDTH = 260;
const MAX_HEIGHT = 420;

interface Anchor {
  x: number;
  y: number;
  width: number;
  height: number;
}

export function useRecipientMenu() {
  const pillRef = useRef<ComponentRef<typeof View>>(null);
  const [anchor, setAnchor] = useState<Anchor | null>(null);
  const window = useWindowDimensions();

  function open() {
    pillRef.current?.measureInWindow((x, y, width, height) => {
      setAnchor({ x, y, width, height });
    });
  }

  function close() {
    setAnchor(null);
  }

  // Just under the pill, centred on it, kept on screen.
  const placement = anchor
    ? {
        top: anchor.y + anchor.height + GAP,
        left: Math.max(
          EDGE,
          Math.min(anchor.x + anchor.width / 2 - MENU_WIDTH / 2, window.width - MENU_WIDTH - EDGE),
        ),
        width: MENU_WIDTH,
        maxHeight: Math.min(
          MAX_HEIGHT,
          window.height - (anchor.y + anchor.height + GAP) - EDGE,
        ),
      }
    : null;

  return { pillRef, placement, isOpen: anchor !== null, open, close };
}
