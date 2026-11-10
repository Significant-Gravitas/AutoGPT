// The base the block menu's rows and chips draw on: the old rounded-md
// Button they were built from, minus the fill, shadow, height and padding
// every one of them sets itself. Each keeps its own rounded-rectangle look.
export const blockMenuButtonBase =
  "inline-flex items-center justify-center whitespace-nowrap rounded-md text-body font-medium leading-5 text-action-foreground transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-focus disabled:pointer-events-none disabled:opacity-50";

export const blockMenuContainerStyle =
  "scrollbar-thin scrollbar-thumb-border-strong scrollbar-track-transparent w-full px-4 pb-4 space-y-3 h-full overflow-y-auto pt-4 transition-all duration-200";

// The drag image is a detached DOM node, so <Text> can't render it; its type
// comes from the role classes below. leading-inherit keeps the line-height it
// inherits from <body>, as the old inline font-size (with no line-height) did.
export const blockDragPreviewClassName =
  "text-body font-medium leading-inherit";

export const blockDragPreviewStyle = `
    padding: 0.625rem 0.875rem;
    background: rgb(var(--color-surface-hover));
    border-radius: 0.75rem;
    color: rgb(var(--color-foreground-emphasis));
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    position: absolute;
    top: -9999px;
  `;
