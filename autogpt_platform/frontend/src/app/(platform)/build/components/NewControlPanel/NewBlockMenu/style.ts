export const blockMenuContainerStyle =
  // TODO(tokens): no fitting role for scrollbar-thumb-zinc-300 #C5C5C9; #C5C5C9 is foreground-disabled (a text colour), border-strong (a border colour)
  "scrollbar-thin scrollbar-thumb-zinc-300 scrollbar-track-transparent w-full px-4 pb-4 space-y-3 h-full overflow-y-auto pt-4 transition-all duration-200";

// The drag image is a detached DOM node, so <Text> can't render it; its type
// comes from the role classes below. leading-inherit keeps the line-height it
// inherits from <body>, as the old inline font-size (with no line-height) did.
export const blockDragPreviewClassName =
  "text-body font-medium leading-inherit";

// TODO(tokens): no exact role: color #27272a #27272A, box-shadow rgba(0, 0, 0, 0.1) #000000; no fitting role for background #f4f4f5 #F4F4F5; #F4F4F5 is muted, whose .dark value (39 39 42) differs (QUALITY 43, SECRT-3154)
export const blockDragPreviewStyle = `
    padding: 0.625rem 0.875rem;
    background: #f4f4f5;
    border-radius: 0.75rem;
    color: #27272a;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    position: absolute;
    top: -9999px;
  `;
