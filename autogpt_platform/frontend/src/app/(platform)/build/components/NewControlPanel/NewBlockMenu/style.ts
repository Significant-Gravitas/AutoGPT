export const blockMenuContainerStyle =
  // TODO(tokens): no fitting role for scrollbar-thumb-zinc-300 #C5C5C9; #C5C5C9 is foreground-disabled (a text colour), border-strong (a border colour)
  "scrollbar-thin scrollbar-thumb-zinc-300 scrollbar-track-transparent w-full px-4 pb-4 space-y-3 h-full overflow-y-auto pt-4 transition-all duration-200";

// TODO(tokens): no exact role: color #27272a #27272A, box-shadow rgba(0, 0, 0, 0.1) #000000
export const blockDragPreviewStyle = `
    padding: 0.625rem 0.875rem;
    background: rgb(var(--color-muted));
    border-radius: 0.75rem;
    font-size: 0.875rem;
    font-weight: 500;
    color: #27272a;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    position: absolute;
    top: -9999px;
  `;
