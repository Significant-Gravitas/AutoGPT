"use client";

import type { ReactNode } from "react";
import type { OutputMetadata, OutputRenderer } from "../../types";
import {
  canRenderNotebook,
  getCopyContentNotebook,
  getDownloadContentNotebook,
  isConcatenableNotebook,
  parseNotebook,
} from "./helpers";
import { NotebookViewer } from "./NotebookViewer";

function renderNotebook(value: unknown, _metadata?: OutputMetadata): ReactNode {
  const notebook = parseNotebook(value);
  if (!notebook) return null;
  return <NotebookViewer notebook={notebook} />;
}

export const notebookRenderer: OutputRenderer = {
  name: "NotebookRenderer",
  priority: 36,
  canRender: canRenderNotebook,
  render: renderNotebook,
  getCopyContent: getCopyContentNotebook,
  getDownloadContent: getDownloadContentNotebook,
  isConcatenable: isConcatenableNotebook,
};
