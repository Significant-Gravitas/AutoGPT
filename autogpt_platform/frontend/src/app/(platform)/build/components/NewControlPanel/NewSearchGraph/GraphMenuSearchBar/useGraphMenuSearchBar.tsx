import { useState, useMemo, useDeferredValue } from "react";
import { CustomNode } from "@/app/(platform)/build/components/legacy-builder/CustomNode/CustomNode";
import { beautifyString } from "@/lib/utils";
import jaro from "jaro-winkler";

export type SearchableNode = CustomNode & {
  searchScore?: number;
  matchedFields?: string[];
};

export const useGraphSearch = (nodes: CustomNode[]) => {
  const [open, setOpen] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");
  const deferredSearchQuery = useDeferredValue(searchQuery);

  const filteredNodes = useMemo(() => {
    // Filter out invalid nodes
    const validNodes = (nodes || []).filter((node) => node && node.data);

    if (!deferredSearchQuery.trim()) {
      return validNodes.map((node) => ({
        ...node,
        searchScore: 1,
        matchedFields: [],
      }));
    }

    const query = deferredSearchQuery.toLowerCase().trim();
    const queryWords = query.split(/\s+/);

    return validNodes
      .map((node): SearchableNode => {
        const { score, matchedFields } = calculateNodeScore(
          node,
          query,
          queryWords,
        );
        return { ...node, searchScore: score, matchedFields };
      })
      .filter((node) => node.searchScore! > 0)
      .sort((a, b) => b.searchScore! - a.searchScore!);
  }, [nodes, deferredSearchQuery]);

  return {
    open,
    setOpen,
    searchQuery,
    setSearchQuery,
    filteredNodes,
  };
};

function calculateNodeScore(
  node: CustomNode,
  query: string,
  queryWords: string[],
): { score: number; matchedFields: string[] } {
  const matchedFields: string[] = [];
  let score = 0;

  // Safety check for node data
  if (!node || !node.data) {
    return { score: 0, matchedFields: [] };
  }

  // Prepare searchable text with defensive checks
  const nodeTitle = (node.data?.title || "").toLowerCase(); // This includes the ID
  const nodeId = (node.id || "").toLowerCase();
  const nodeDescription = (node.data?.description || "").toLowerCase();
  const blockType = (node.data?.blockType || "").toLowerCase();
  const beautifiedBlockType = beautifyString(blockType).toLowerCase();
  const customizedName = (
    node.data?.metadata?.customized_name || ""
  ).toLowerCase();

  // Get input and output names with defensive checks
  const inputNames = Object.keys(node.data?.inputSchema?.properties || {}).map(
    (key) => key.toLowerCase(),
  );
  const outputNames = Object.keys(
    node.data?.outputSchema?.properties || {},
  ).map((key) => key.toLowerCase());

  // 1. Check exact match in customized name, title (includes ID), node ID, or block type (highest priority)
  if (
    customizedName.includes(query) ||
    nodeTitle.includes(query) ||
    nodeId.includes(query) ||
    blockType.includes(query) ||
    beautifiedBlockType.includes(query)
  ) {
    score = 4;
    matchedFields.push("title");
  }

  // 2. Check all query words in customized name, title or block type
  else if (
    queryWords.every(
      (word) =>
        customizedName.includes(word) ||
        nodeTitle.includes(word) ||
        beautifiedBlockType.includes(word),
    )
  ) {
    score = 3.5;
    matchedFields.push("title");
  }

  // 3. Check exact match in input/output names
  else if (inputNames.some((name) => name.includes(query))) {
    score = 3;
    matchedFields.push("inputs");
  } else if (outputNames.some((name) => name.includes(query))) {
    score = 2.8;
    matchedFields.push("outputs");
  }

  // 4. Check all query words in input/output names
  else if (
    inputNames.some((name) => queryWords.every((word) => name.includes(word)))
  ) {
    score = 2.5;
    matchedFields.push("inputs");
  } else if (
    outputNames.some((name) => queryWords.every((word) => name.includes(word)))
  ) {
    score = 2.3;
    matchedFields.push("outputs");
  }

  // 5. Similarity matching using Jaro-Winkler
  else {
    const titleSimilarity = Math.max(
      jaro(customizedName, query),
      jaro(nodeTitle, query),
      jaro(nodeId, query),
      jaro(beautifiedBlockType, query),
    );

    if (titleSimilarity > 0.7) {
      score = 1.5 + titleSimilarity;
      matchedFields.push("title");
    }

    // Check similarity with input/output names
    const inputSimilarity = Math.max(
      ...inputNames.map((name) => jaro(name, query)),
      0,
    );
    const outputSimilarity = Math.max(
      ...outputNames.map((name) => jaro(name, query)),
      0,
    );

    if (inputSimilarity > 0.7 && inputSimilarity > score) {
      score = 1 + inputSimilarity;
      matchedFields.push("inputs");
    }
    if (outputSimilarity > 0.7 && outputSimilarity > score) {
      score = 0.8 + outputSimilarity;
      matchedFields.push("outputs");
    }
  }

  // 6. Check description (lower priority)
  if (score === 0 && nodeDescription.includes(query)) {
    score = 0.5;
    matchedFields.push("description");
  }

  // 7. Check if all query words appear in description
  if (
    score === 0 &&
    queryWords.every((word) => nodeDescription.includes(word))
  ) {
    score = 0.3;
    matchedFields.push("description");
  }

  return { score, matchedFields };
}
