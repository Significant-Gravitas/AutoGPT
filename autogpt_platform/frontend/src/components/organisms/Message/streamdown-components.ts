import type { StreamdownProps } from "streamdown";

export type StreamdownComponents = NonNullable<StreamdownProps["components"]>;

function hasSameEntries(a: StreamdownComponents, b: StreamdownComponents) {
  const keys = Object.keys(a) as (keyof StreamdownComponents)[];
  return (
    keys.length === Object.keys(b).length && keys.every((k) => a[k] === b[k])
  );
}

// Streamdown re-parses every block of a message when the identity of a
// component in `components` changes, and React remounts any element whose
// component type changes. This builds `merge(components)` once per
// `components` object, and again only if one of its entries has changed
// since, so a caller that edits its map in place still gets the new renderer.
export function createComponentsMerger(
  merge: (components: StreamdownComponents) => StreamdownComponents,
) {
  const cache = new WeakMap<
    StreamdownComponents,
    { snapshot: StreamdownComponents; merged: StreamdownComponents }
  >();
  return function mergeOnce(components: StreamdownComponents) {
    const cached = cache.get(components);
    if (cached && hasSameEntries(cached.snapshot, components))
      return cached.merged;
    const merged = merge(components);
    cache.set(components, { snapshot: { ...components }, merged });
    return merged;
  };
}
