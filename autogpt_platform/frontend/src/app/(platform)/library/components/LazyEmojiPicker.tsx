"use client";

import { EmojiPicker } from "@ferrucc-io/emoji-picker";

interface Props {
  onEmojiSelect: (emoji: string) => void;
  containerHeight?: number;
}

export function LazyEmojiPicker({
  onEmojiSelect,
  containerHeight = 295,
}: Props) {
  return (
    <EmojiPicker
      onEmojiSelect={onEmojiSelect}
      emojiSize={32}
      className="w-full rounded-2xl px-2"
    >
      <EmojiPicker.Group>
        <EmojiPicker.List hideStickyHeader containerHeight={containerHeight} />
      </EmojiPicker.Group>
    </EmojiPicker>
  );
}
