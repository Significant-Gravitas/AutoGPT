import { Graph } from "@/app/api/__generated__/models/graph";

/** Sanitizes a graph object in place so it can "safely" be imported into the system.
 *
 * **⚠️ Note:** not an actual safety feature, just intended to make the import UX more reliable.
 */
export function sanitizeImportedGraph(graph: Graph): void {
  updateBlockIDs(graph);
  removeCredentials(graph);
}

/** Removes (in place) the credentials the nodes still hold, sub-workflows'
 * nodes included: each credentials input that points at a stored credential,
 * which would block the save or point at someone else's. The file carries no
 * block schemas, so the input's name marks these inputs: blocks must name every
 * credentials field `credentials` or `*_credentials`. Anything else is the
 * user's data, even when it is keyed "credentials" or looks like a credential. */
function removeCredentials(graph: Graph): void {
  const nodes = [graph, ...(graph.sub_graphs ?? [])].flatMap(
    (g) => g.nodes ?? [],
  );
  for (const node of nodes) {
    const inputs = node.input_default ?? {};
    for (const [name, value] of Object.entries(inputs)) {
      if (isCredentialsFieldName(name) && pointsAtStoredCredential(value)) {
        delete inputs[name];
      }
    }
  }
}

function isCredentialsFieldName(name: string): boolean {
  return name === "credentials" || name.endsWith("_credentials");
}

/** A credentials value with an `id` names a stored credential; one without
 * points at nothing, so the save skips it. */
function pointsAtStoredCredential(value: unknown): boolean {
  if (typeof value !== "object" || value === null) return false;
  return typeof (value as Record<string, unknown>).id === "string";
}

/** ⚠️ Remove after 2025-10-01 (one year after implementation in
 * [#8229](https://github.com/Significant-Gravitas/AutoGPT/pull/8229))
 */
function updateBlockIDs(graph: Graph) {
  graph.nodes
    ?.filter((node) => node.block_id in updatedBlockIDMap)
    .forEach((node) => {
      node.block_id = updatedBlockIDMap[node.block_id];
    });
}

const updatedBlockIDMap: Record<string, string> = {
  // https://github.com/Significant-Gravitas/AutoGPT/issues/8223
  "a1b2c3d4-5e6f-7g8h-9i0j-k1l2m3n4o5p6":
    "436c3984-57fd-4b85-8e9a-459b356883bd",
  "b2g2c3d4-5e6f-7g8h-9i0j-k1l2m3n4o5p6":
    "0e50422c-6dee-4145-83d6-3a5a392f65de",
  "c3d4e5f6-7g8h-9i0j-1k2l-m3n4o5p6q7r8":
    "a0a69be1-4528-491c-a85a-a4ab6873e3f0",
  "c3d4e5f6-g7h8-i9j0-k1l2-m3n4o5p6q7r8":
    "32a87eab-381e-4dd4-bdb8-4c47151be35a",
  "b2c3d4e5-6f7g-8h9i-0j1k-l2m3n4o5p6q7":
    "87840993-2053-44b7-8da4-187ad4ee518c",
  "h1i2j3k4-5l6m-7n8o-9p0q-r1s2t3u4v5w6":
    "d0822ab5-9f8a-44a3-8971-531dd0178b6b",
  "d3f4g5h6-1i2j-3k4l-5m6n-7o8p9q0r1s2t":
    "df06086a-d5ac-4abb-9996-2ad0acb2eff7",
  "h5e7f8g9-1b2c-3d4e-5f6g-7h8i9j0k1l2m":
    "f5b0f5d0-1862-4d61-94be-3ad0fa772760",
  "a1234567-89ab-cdef-0123-456789abcdef":
    "4335878a-394e-4e67-adf2-919877ff49ae",
  "f8e7d6c5-b4a3-2c1d-0e9f-8g7h6i5j4k3l":
    "f66a3543-28d3-4ab5-8945-9b336371e2ce",
  "b29c1b50-5d0e-4d9f-8f9d-1b0e6fcbf0h2":
    "716a67b3-6760-42e7-86dc-18645c6e00fc",
  "31d1064e-7446-4693-o7d4-65e5ca9110d1":
    "cc10ff7b-7753-4ff2-9af6-9399b1a7eddc",
  "c6731acb-4105-4zp1-bc9b-03d0036h370g":
    "5ebe6768-8e5d-41e3-9134-1c7bd89a8d52",
};
