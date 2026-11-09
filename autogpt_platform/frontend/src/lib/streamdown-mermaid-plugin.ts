import type { DiagramPlugin } from "streamdown";

type MermaidConfig = Parameters<DiagramPlugin["getMermaid"]>[0];

let pluginPromise: Promise<DiagramPlugin> | null = null;

// @streamdown/mermaid imports all of mermaid at the top of its module, which put
// the library in every chat's first load. It is fetched the first time a reply
// actually draws a diagram, and Streamdown only ever awaits `render`.
function loadMermaidPlugin(): Promise<DiagramPlugin> {
  if (!pluginPromise) {
    pluginPromise = import("@streamdown/mermaid").then(
      (module) => module.mermaid,
      (error: unknown) => {
        pluginPromise = null;
        throw error;
      },
    );
  }
  return pluginPromise;
}

export const mermaid: DiagramPlugin = {
  name: "mermaid",
  type: "diagram",
  language: "mermaid",
  getMermaid(config?: MermaidConfig) {
    let pendingConfig = config;
    return {
      initialize(nextConfig) {
        pendingConfig = nextConfig;
      },
      async render(id, source) {
        const plugin = await loadMermaidPlugin();
        return plugin.getMermaid(pendingConfig).render(id, source);
      },
    };
  },
};
