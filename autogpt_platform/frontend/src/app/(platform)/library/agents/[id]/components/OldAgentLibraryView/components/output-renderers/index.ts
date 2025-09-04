import { globalRegistry } from "./types";
import { textRenderer } from "./renderers/TextRenderer";
import { codeRenderer } from "./renderers/CodeRenderer";
import { imageRenderer } from "./renderers/ImageRenderer";
import { videoRenderer } from "./renderers/VideoRenderer";
import { jsonRenderer } from "./renderers/JSONRenderer";
import { markdownRenderer } from "./renderers/MarkdownRenderer";

// Register all renderers in priority order
globalRegistry.register(videoRenderer);
globalRegistry.register(imageRenderer);
globalRegistry.register(codeRenderer);
globalRegistry.register(markdownRenderer);
globalRegistry.register(jsonRenderer);
globalRegistry.register(textRenderer);

export { globalRegistry };
export type { OutputRenderer, OutputMetadata, DownloadContent } from "./types";
export { OutputItem } from "./components/OutputItem";
export { OutputActions } from "./components/OutputActions";
