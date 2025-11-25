export const TUTORIAL_SELECTORS = {
  BLOCKS_TRIGGER: '[data-id="blocks-control-popover-trigger"]',
  BLOCKS_CONTENT: '[data-id="blocks-control-popover-content"]',
  BLOCKS_SEARCH_INPUT:
    '[data-id="blocks-control-popover-content"] input[type="text"]',
  FIT_VIEW_BUTTON: ".react-flow__controls-fitview",
  BLOCK_CARD_PREFIX: '[data-id^="block-card-"]',
} as const;

export const CSS_CLASSES = {
  DISABLE: "disable-blocks",
  HIGHLIGHT: "highlight-block",
} as const;

export const TUTORIAL_CONFIG = {
  ELEMENT_CHECK_INTERVAL: 10, // ms
  USE_MODAL_OVERLAY: true,
  SCROLL_BEHAVIOR: "smooth" as const,
  SCROLL_BLOCK: "center" as const,
} as const;
