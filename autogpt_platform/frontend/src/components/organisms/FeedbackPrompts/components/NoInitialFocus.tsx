// The dialog moves focus to its first tabbable element on open, which is the
// close button. Taking focus back onto this untabbable span right after keeps
// the focus ring off it.
export function NoInitialFocus() {
  return (
    <span
      tabIndex={-1}
      className="sr-only"
      ref={(el) => {
        if (el) requestAnimationFrame(() => el.focus());
      }}
    />
  );
}
