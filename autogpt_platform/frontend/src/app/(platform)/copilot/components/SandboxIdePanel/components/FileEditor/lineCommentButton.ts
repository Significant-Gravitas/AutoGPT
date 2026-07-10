import {
  Decoration,
  type DecorationSet,
  EditorView,
  type Extension,
  StateEffect,
  StateField,
  WidgetType,
} from "@uiw/react-codemirror";

/** Max lines of code captured in a single reference. */
const MAX_SNIPPET_LINES = 60;

/** Details handed to React when the user clicks a line's "+" button. */
export interface LineCommentRequest {
  fromLine: number;
  toLine: number;
  code: string;
  anchorRect: DOMRect;
}

/** The 1-based line the pointer is currently over, or null when outside. */
const setHoverLine = StateEffect.define<number | null>();

const hoverLineField = StateField.define<number | null>({
  create: () => null,
  update(value, tr) {
    for (const effect of tr.effects) {
      if (effect.is(setHoverLine)) return effect.value;
    }
    return value;
  },
});

class AddCommentWidget extends WidgetType {
  constructor(
    private readonly line: number,
    private readonly onRequest: (
      view: EditorView,
      line: number,
      button: HTMLElement,
    ) => void,
  ) {
    super();
  }

  eq(other: AddCommentWidget) {
    return other.line === this.line;
  }

  toDOM(view: EditorView) {
    const anchor = document.createElement("span");
    anchor.className = "cm-add-comment-anchor";
    const button = document.createElement("button");
    button.type = "button";
    button.className = "cm-add-comment-button";
    button.setAttribute("aria-label", "Comment on this line");
    button.title = "Comment on this line";
    button.textContent = "+";
    button.addEventListener("mousedown", (event) => {
      event.preventDefault();
      event.stopPropagation();
      this.onRequest(view, this.line, button);
    });
    anchor.appendChild(button);
    return anchor;
  }

  ignoreEvent() {
    return true;
  }
}

interface Args {
  onRequestComment: (request: LineCommentRequest) => void;
}

/**
 * Reveals a primary "+" button on the left of the hovered line. Clicking it
 * captures the line (or the current multi-line selection, if it covers the
 * hovered line) and asks React to open the comment popover next to it.
 */
export function lineCommentButton({ onRequestComment }: Args): Extension {
  function handleRequest(
    view: EditorView,
    clickedLine: number,
    button: HTMLElement,
  ) {
    const { doc, selection } = view.state;
    let fromLine = clickedLine;
    let toLine = clickedLine;

    const range = selection.main;
    if (!range.empty) {
      const selFrom = doc.lineAt(range.from).number;
      const selTo = doc.lineAt(range.to).number;
      if (clickedLine >= selFrom && clickedLine <= selTo) {
        fromLine = selFrom;
        toLine = selTo;
      }
    }

    const end = Math.min(toLine, fromLine + MAX_SNIPPET_LINES - 1);
    const lines: string[] = [];
    for (let n = fromLine; n <= end; n++) lines.push(doc.line(n).text);
    if (end < toLine) lines.push("…");

    onRequestComment({
      fromLine,
      toLine,
      code: lines.join("\n"),
      anchorRect: button.getBoundingClientRect(),
    });
  }

  const decorations = EditorView.decorations.compute(
    [hoverLineField],
    (state) => {
      const line = state.field(hoverLineField);
      if (line == null || line < 1 || line > state.doc.lines) {
        return Decoration.none as DecorationSet;
      }
      const lineObj = state.doc.line(line);
      const widget = Decoration.widget({
        widget: new AddCommentWidget(line, handleRequest),
        side: -1,
      });
      return Decoration.set([widget.range(lineObj.from)]);
    },
  );

  return [
    hoverLineField,
    decorations,
    EditorView.domEventHandlers({
      mousemove(event, view) {
        const target = event.target as HTMLElement | null;
        if (target?.closest(".cm-add-comment-button")) return false;
        const pos = view.posAtCoords({ x: event.clientX, y: event.clientY });
        if (pos == null) return false;
        const line = view.state.doc.lineAt(pos).number;
        if (view.state.field(hoverLineField) !== line) {
          view.dispatch({ effects: setHoverLine.of(line) });
        }
        return false;
      },
      mouseleave(_event, view) {
        if (view.state.field(hoverLineField) !== null) {
          view.dispatch({ effects: setHoverLine.of(null) });
        }
        return false;
      },
    }),
  ];
}
