import { globalRegistry } from "./types";
import { TextRenderer } from "./renderers/TextRenderer";
import { CodeRenderer } from "./renderers/CodeRenderer";
import { ImageRenderer } from "./renderers/ImageRenderer";
import { VideoRenderer } from "./renderers/VideoRenderer";
import { JSONRenderer } from "./renderers/JSONRenderer";

// Register all renderers in priority order
globalRegistry.register(new VideoRenderer());
globalRegistry.register(new ImageRenderer());
globalRegistry.register(new CodeRenderer());
globalRegistry.register(new JSONRenderer());
globalRegistry.register(new TextRenderer());

export { globalRegistry };
export type { OutputRenderer, OutputMetadata, DownloadContent } from "./types";
export { OutputItem } from "./components/OutputItem";
export { OutputActions } from "./components/OutputActions";
