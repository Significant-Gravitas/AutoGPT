// No version in the link: the builder opens the workflow's active version, so
// a page that cached an older version can't send the user back to it.
export function getBuilderHref(graphID: string) {
  return `/build?flowID=${encodeURIComponent(graphID)}`;
}
