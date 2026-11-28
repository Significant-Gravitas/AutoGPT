import type { FlashListRef } from "@shopify/flash-list";
import { useRef, useState } from "react";
import { Keyboard, type LayoutChangeEvent } from "react-native";
import { useDerivedValue, useSharedValue } from "react-native-reanimated";

import type { Turn } from "./turns";

/**
 * The thread's scroll, the way ChatGPT does it: a sent message scrolls to the
 * top of the screen and its reply fills the space below, without the view
 * chasing the stream. Until the first send here, the list keeps to its end.
 * `topInset` is the header floating over the list's top, which the sent
 * message stops below.
 */
export function useMessageList(topInset: number) {
  const listRef = useRef<FlashListRef<Turn>>(null);
  const viewport = useRef(0);
  const lastTurnHeight = useRef(0);
  // Turns before the send; the next turn past this count scrolls up.
  const pendingAnchor = useRef<number | null>(null);
  const [isAnchored, setIsAnchored] = useState(false);
  const [isAtEnd, setIsAtEnd] = useState(true);
  // The room under the last turn before the header's inset; null while
  // unanchored. The inset is applied below, so a header that changes height
  // updates the blank space without a new layout pass.
  const roomBelow = useSharedValue<number | null>(null);
  const blankSpace = useDerivedValue(() =>
    roomBelow.value === null ? 0 : Math.max(0, roomBelow.value - topInset),
  );
  const composerHeight = useSharedValue(0);

  function updateBlank(anchored: boolean) {
    roomBelow.value = anchored
      ? viewport.current - lastTurnHeight.current - BOTTOM_PADDING
      : null;
  }

  /** Call once a send is accepted; `turnCount` is the turns shown before it. */
  function anchorNextTurn(turnCount: number) {
    pendingAnchor.current = turnCount;
    setIsAnchored(true);
    Keyboard.dismiss();
  }

  function handleViewportLayout(event: LayoutChangeEvent) {
    viewport.current = event.nativeEvent.layout.height;
    updateBlank(isAnchored || pendingAnchor.current !== null);
  }

  function handleLastTurnLayout(index: number, height: number) {
    lastTurnHeight.current = height;
    const pending = pendingAnchor.current;
    const isNewTurn = pending !== null && index >= pending;
    updateBlank(isAnchored || isNewTurn);
    if (!isNewTurn) return;
    pendingAnchor.current = null;
    // The inset lands on the UI thread; wait for it before scrolling.
    requestAnimationFrame(() =>
      requestAnimationFrame(() => {
        void listRef.current?.scrollToEnd({ animated: true });
      }),
    );
  }

  function handleComposerLayout(event: LayoutChangeEvent) {
    composerHeight.value = event.nativeEvent.layout.height;
  }

  function scrollToEnd() {
    void listRef.current?.scrollToEnd({ animated: true });
  }

  function jumpToEnd() {
    void listRef.current?.scrollToEnd({ animated: false });
  }

  return {
    listRef,
    blankSpace,
    composerHeight,
    isAnchored,
    isAtEnd,
    setIsAtEnd,
    anchorNextTurn,
    handleViewportLayout,
    handleLastTurnLayout,
    handleComposerLayout,
    scrollToEnd,
    jumpToEnd,
  };
}

/** The list's padding under the last turn. */
export const BOTTOM_PADDING = 16;
