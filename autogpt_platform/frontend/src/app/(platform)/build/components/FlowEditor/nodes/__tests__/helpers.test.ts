import { describe, it, expect } from "vitest";
import { augmentOutputSchemaWithEdges } from "../helpers";

describe("augmentOutputSchemaWithEdges", () => {
  it("returns empty properties when outputSchema has no properties", () => {
    const outputSchema = {};
    const edges: any[] = [];
    const result = augmentOutputSchemaWithEdges(outputSchema, "node-a", edges);
    expect(result).toEqual({});
  });

  it("does not modify schema when there are no matching outgoing edges", () => {
    const outputSchema = {
      properties: {
        response: { type: "object" },
      },
    };
    const edges = [
      { source: "node-b", sourceHandle: "response_#_key" }, // different node
      { source: "node-a", sourceHandle: "other" }, // non-nested handle
    ];
    const result = augmentOutputSchemaWithEdges(outputSchema, "node-a", edges);
    expect(result).toEqual({
      response: { type: "object" },
    });
  });

  it("dynamically injects nested properties for matching outgoing edges", () => {
    const outputSchema = {
      properties: {
        response: { type: "object" },
      },
    };
    const edges = [
      { source: "node-a", sourceHandle: "response_#_has_new_reviews" },
      { source: "node-a", sourceHandle: "response_#_reviews_#_author" },
    ];
    const result = augmentOutputSchemaWithEdges(outputSchema, "node-a", edges);
    expect(result).toEqual({
      response: {
        type: "object",
        properties: {
          has_new_reviews: { title: "has_new_reviews" },
          reviews: {
            type: "object",
            title: "reviews",
            properties: {
              author: { title: "author" },
            },
          },
        },
      },
    });
  });
});
