export const AGENT_LIBRARY_SECTION_PADDING_X = "px-4";

export function getLibraryAgentBuilderHref(graphId: string) {
  return `/build?flowID=${graphId}`;
}
