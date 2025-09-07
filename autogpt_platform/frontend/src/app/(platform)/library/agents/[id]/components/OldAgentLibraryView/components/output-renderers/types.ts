import { ReactNode } from "react";

export interface OutputMetadata {
  type?: string;
  language?: string;
  mimeType?: string;
  filename?: string;
  [key: string]: any;
}

export interface DownloadContent {
  data: Blob | string;
  filename: string;
  mimeType: string;
}

export interface CopyContent {
  mimeType: string; // Primary MIME type to try
  data: Blob | string | (() => Promise<Blob | string>); // Data or async function to get data
  fallbackText?: string; // Optional fallback text if rich copy fails
  alternativeMimeTypes?: string[]; // Alternative MIME types to try if primary isn't supported
}

export interface OutputRenderer {
  name: string;
  priority: number;
  canRender(value: any, metadata?: OutputMetadata): boolean;
  render(value: any, metadata?: OutputMetadata): ReactNode;
  getCopyContent(value: any, metadata?: OutputMetadata): CopyContent | null;
  getDownloadContent(
    value: any,
    metadata?: OutputMetadata,
  ): DownloadContent | null;
  isConcatenable(value: any, metadata?: OutputMetadata): boolean;
}

export class OutputRendererRegistry {
  private renderers: OutputRenderer[] = [];

  register(renderer: OutputRenderer): void {
    const index = this.renderers.findIndex(
      (r) => r.priority < renderer.priority,
    );
    if (index === -1) {
      this.renderers.push(renderer);
    } else {
      this.renderers.splice(index, 0, renderer);
    }
  }

  getRenderer(value: any, metadata?: OutputMetadata): OutputRenderer | null {
    return this.renderers.find((r) => r.canRender(value, metadata)) || null;
  }

  getAllRenderers(): OutputRenderer[] {
    return [...this.renderers];
  }
}

export const globalRegistry = new OutputRendererRegistry();
