import { globalRegistry } from "./types";
import { textRenderer } from "./renderers/TextRenderer";
import { codeRenderer } from "./renderers/CodeRenderer";
import { csvRenderer } from "./renderers/CSVRenderer";
import { htmlRenderer } from "./renderers/HTMLRenderer";
import { imageRenderer } from "./renderers/ImageRenderer";
import { videoRenderer } from "./renderers/VideoRenderer";
import { audioRenderer } from "./renderers/AudioRenderer";
import { jsonRenderer } from "./renderers/JSONRenderer";
import { markdownRenderer } from "./renderers/MarkdownRenderer";
import { workspaceFileRenderer } from "./renderers/WorkspaceFileRenderer";
import { linkRenderer } from "./renderers/LinkRenderer";

// Register all renderers in priority order
globalRegistry.register(workspaceFileRenderer);
globalRegistry.register(videoRenderer);
globalRegistry.register(audioRenderer);
globalRegistry.register(htmlRenderer);
globalRegistry.register(imageRenderer);
globalRegistry.register(csvRenderer);
globalRegistry.register(codeRenderer);
globalRegistry.register(markdownRenderer);
globalRegistry.register(jsonRenderer);
globalRegistry.register(linkRenderer);
globalRegistry.register(textRenderer);

export { globalRegistry };
export type { OutputRenderer, OutputMetadata, DownloadContent } from "./types";
export { OutputItem } from "./components/OutputItem";
export { OutputActions } from "./components/OutputActions";
