import { describe, it, expect, afterEach } from "vitest";
import {
  augmentOutputSchemaWithEdges,
  getNestedOutputProperties,
} from "../outputSchema";
import type { RJSFSchema } from "@rjsf/utils";
import type { CustomEdge } from "../../edges/CustomEdge";

afterEach(() => {
  for (const target of [
    Object.prototype,
    Object,
    Object.prototype.toString,
    Object.prototype.hasOwnProperty,
  ]) {
    Reflect.deleteProperty(target, "properties");
  }
});

describe("augmentOutputSchemaWithEdges", () => {
  it("returns empty properties when outputSchema has no properties", () => {
    const outputSchema: RJSFSchema = {};
    const edges: Pick<CustomEdge, "source" | "sourceHandle">[] = [];
    const result = augmentOutputSchemaWithEdges(outputSchema, "node-a", edges);
    expect(result).toEqual({});
  });

  it("does not modify schema when there are no matching outgoing edges", () => {
    const outputSchema: RJSFSchema = {
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
    const outputSchema: RJSFSchema = {
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

describe("dynamic output safety", () => {
  function augment(schema: RJSFSchema, ...handles: string[]) {
    return augmentOutputSchemaWithEdges(
      schema,
      "node-a",
      handles.map((sourceHandle) => ({ source: "node-a", sourceHandle })),
    );
  }

  it.each([
    "__proto__",
    "constructor",
    "prototype",
    "toString",
    "hasOwnProperty",
  ])(
    "treats %s as an own dictionary key without changing prototypes",
    (key) => {
      const before = Object.getOwnPropertyDescriptors(Object.prototype);
      try {
        const result = augment(
          { properties: { response: { type: "object" } } },
          `response_#_${key}_#_injected`,
        );
        expect(Object.getOwnPropertyDescriptors(Object.prototype)).toEqual(
          before,
        );
        const children = (result.response as RJSFSchema).properties!;
        expect(Object.hasOwn(children, key)).toBe(true);
        expect((children[key] as RJSFSchema).properties?.injected).toEqual({
          title: "injected",
        });
      } finally {
        Reflect.deleteProperty(Object.prototype, "properties");
        Reflect.deleteProperty(Object, "properties");
        Reflect.deleteProperty(Object.prototype.toString, "properties");
        Reflect.deleteProperty(Object.prototype.hasOwnProperty, "properties");
      }
    },
  );

  it("preserves declared prototype keys and never changes the source schema", () => {
    const schema = JSON.parse(
      '{"properties":{"__proto__":{"type":"object","properties":{"constructor":{"type":"object"}}}}}',
    ) as RJSFSchema;
    const before = JSON.stringify(schema);
    const result = augment(schema, "__proto___#_constructor_#_name");
    expect(Object.hasOwn(result, "__proto__")).toBe(true);
    expect(JSON.stringify(schema)).toBe(before);
    expect((result.__proto__ as RJSFSchema).properties?.constructor).toEqual({
      type: "object",
      properties: { name: { title: "name" } },
    });
  });

  it("does not invent missing top-level outputs or empty path segments", () => {
    const schema: RJSFSchema = { properties: { response: { type: "object" } } };
    expect(
      augment(
        schema,
        "missing_#_child",
        "__proto___#_child",
        "response_#_",
        "response_#__#_child",
        "response" + "_#_deep".repeat(10000),
      ),
    ).toEqual(schema.properties);
  });

  it.each(
    [
      false,
      null,
      1,
      "bad",
      [],
      { type: "string" },
      { type: "boolean" },
      { type: "array" },
      { type: "object", additionalProperties: false },
      { anyOf: [{ type: "string" }, { type: "null" }] },
    ].map((value) => [value]),
  )("does not descend into a closed, scalar or invalid schema: %j", (field) => {
    const schema = {
      properties: { response: field },
    } as unknown as RJSFSchema;
    const result = augment(schema, "response_#_child_#_name");
    expect(result).toEqual(schema.properties);
  });

  it("does not traverse inherited root or nested properties", () => {
    const inherited = { injected: { type: "object", properties: {} } };
    const schema: RJSFSchema = {
      properties: {
        response: { type: "object", properties: Object.create(inherited) },
      },
    };
    const result = augment(schema, "response_#_injected_#_child");
    expect(inherited.injected.properties).toEqual({});
    expect(
      Object.hasOwn((result.response as RJSFSchema).properties!, "injected"),
    ).toBe(true);
    expect(
      augment(
        { properties: Object.create({ response: { type: "object" } }) },
        "response_#_child",
      ),
    ).toEqual({});
  });

  it("supports nullable objects while keeping typed dictionary values", () => {
    const schema: RJSFSchema = {
      properties: {
        response: {
          title: "Response",
          anyOf: [
            { type: "null" },
            { type: "object", additionalProperties: { type: "string" } },
          ],
        },
      },
    };
    expect(
      augment(schema, "response_#_name", "response_#_second").response,
    ).toMatchObject({
      title: "Response",
      properties: {
        name: { title: "name", type: "string" },
        second: { title: "second", type: "string" },
      },
    });
    expect(augment(schema, "response_#_name_#_invalid")).toEqual(
      schema.properties,
    );
  });

  it.each(
    [
      undefined,
      null,
      true,
      false,
      [],
      "bad",
      { properties: null },
      { properties: [] },
      { properties: "bad" },
    ].map((value) => [value]),
  )("handles a missing or invalid root schema: %j", (schema) => {
    expect(
      augment(schema as unknown as RJSFSchema, "response_#_child"),
    ).toEqual({});
  });

  it("expands true and Any schemas without losing metadata or depending on edge order", () => {
    const schema: RJSFSchema = {
      properties: {
        response: {
          title: "Response",
          properties: { allowed: true, denied: false },
        },
      },
    };
    const handles = [
      "response_#_reviews",
      "response_#_reviews_#_author",
      "response_#_allowed_#_name",
      "response_#_denied_#_name",
    ];
    const result = augment(schema, ...handles);
    expect(result).toEqual(augment(schema, ...handles.toReversed()));
    expect(result.response).toMatchObject({
      type: "object",
      title: "Response",
      properties: {
        reviews: {
          type: "object",
          properties: { author: { title: "author" } },
        },
        allowed: { type: "object", properties: { name: { title: "name" } } },
        denied: false,
      },
    });
    expect(schema.properties?.response).not.toHaveProperty("type");
  });

  it("uses the declared additionalProperties schema and preserves existing fields", () => {
    const schema: RJSFSchema = {
      properties: {
        response: {
          type: "object",
          additionalProperties: { type: "string" },
          properties: { known: { type: "object", description: "Keep me" } },
        },
      },
    };
    const result = augment(
      schema,
      "response_#_text",
      "response_#_text_#_invalid",
      "response_#_known_#_name",
    );
    expect((result.response as RJSFSchema).properties).toEqual({
      text: { title: "text", type: "string" },
      known: {
        type: "object",
        description: "Keep me",
        properties: { name: { title: "name" } },
      },
    });
  });

  it("reads object members of nullable and union outputs", () => {
    expect(
      getNestedOutputProperties({
        title: "Main Result",
        anyOf: [
          {
            type: "object",
            properties: {
              text: { title: "Text" },
              html: { title: "Html" },
            },
          },
          { type: "null" },
        ],
      }),
    ).toEqual({
      text: { title: "Text" },
      html: { title: "Html" },
    });
    expect(
      getNestedOutputProperties({
        oneOf: [
          { type: "object", properties: { url: { type: "string" } } },
          { type: "object", properties: { profile: { type: "object" } } },
        ],
      }),
    ).toEqual({
      url: { type: "string" },
      profile: { type: "object" },
    });
    expect(
      getNestedOutputProperties({
        allOf: [{ type: "object", properties: { kept: { title: "Kept" } } }],
      }),
    ).toEqual({ kept: { title: "Kept" } });
  });

  it("ignores scalar unions and keeps declared properties ahead of combinators", () => {
    expect(
      getNestedOutputProperties({
        anyOf: [{ type: "string" }, { type: "null" }],
      }),
    ).toEqual({});
    expect(
      getNestedOutputProperties({
        properties: { kept: { title: "Kept" } },
        anyOf: [
          { type: "object", properties: { ignored: { title: "Ignored" } } },
        ],
      }),
    ).toEqual({ kept: { title: "Kept" } });
    expect(
      getNestedOutputProperties({
        properties: { kept: { title: "Kept" } },
        oneOf: [
          { type: "object", properties: { ignored: { title: "Ignored" } } },
        ],
      }),
    ).toEqual({ kept: { title: "Kept" } });
  });

  it("merges additive allOf properties with direct properties", () => {
    const schema = {
      type: "object" as const,
      properties: {
        kept: { title: "Kept" },
        profile: {
          type: "object" as const,
          title: "Profile",
          properties: { name: { type: "string", title: "Name" } },
        },
      },
      allOf: [
        {
          type: "object" as const,
          properties: {
            added: { title: "Added" },
            kept: { title: "From allOf" },
            profile: {
              type: "object" as const,
              properties: { email: { type: "string", title: "Email" } },
            },
          },
        },
      ],
      anyOf: [
        {
          type: "object" as const,
          properties: { ignored: { title: "Ignored" } },
        },
      ],
      oneOf: [
        {
          type: "object" as const,
          properties: { alsoIgnored: { title: "Also ignored" } },
        },
      ],
    };
    const before = JSON.stringify(schema);
    expect(getNestedOutputProperties(schema)).toEqual({
      kept: { title: "Kept" },
      profile: {
        type: "object",
        title: "Profile",
        properties: {
          name: { type: "string", title: "Name" },
          email: { type: "string", title: "Email" },
        },
      },
      added: { title: "Added" },
    });
    expect(JSON.stringify(schema)).toBe(before);
  });

  it("keeps allOf-only fields visible after a direct field is connected", () => {
    const expanded = augment(
      {
        properties: {
          response: {
            type: "object",
            title: "Response",
            properties: { text: { type: "string", title: "Text" } },
            allOf: [
              {
                type: "object",
                properties: { html: { type: "string", title: "Html" } },
              },
            ],
          },
        },
      },
      "response_#_text",
    );
    expect(getNestedOutputProperties(expanded.response)).toEqual({
      text: { type: "string", title: "Text" },
      html: { type: "string", title: "Html" },
    });
  });

  it("keeps sibling fields when a oneOf or allOf field is connected", () => {
    const siblings = {
      text: { type: "string", title: "Text" },
      html: { type: "string", title: "Html" },
    };
    for (const keyword of ["oneOf", "allOf"] as const) {
      const result = augment(
        {
          properties: {
            response: {
              title: "Response",
              [keyword]: [
                { type: "null" },
                { type: "object", properties: siblings },
              ],
            },
          },
        },
        "response_#_html",
      );
      expect((result.response as RJSFSchema).properties).toEqual(siblings);
    }
  });

  it("resolves a connected field from a later anyOf alternative", () => {
    const result = augment(
      {
        properties: {
          response: {
            anyOf: [
              {
                type: "object",
                properties: { url: { type: "string", title: "Url" } },
              },
              {
                type: "object",
                properties: {
                  profile: {
                    type: "object",
                    title: "Profile",
                    properties: {
                      first_name: { type: "string", title: "First Name" },
                      last_name: { type: "string", title: "Last Name" },
                    },
                  },
                  name: { type: "string", title: "Name" },
                },
              },
            ],
          },
        },
      },
      "response_#_profile_#_first_name",
    );
    expect((result.response as RJSFSchema).properties).toEqual({
      profile: {
        type: "object",
        title: "Profile",
        properties: {
          first_name: { type: "string", title: "First Name" },
          last_name: { type: "string", title: "Last Name" },
        },
      },
      name: { type: "string", title: "Name" },
    });
  });

  it("merges nested properties when union branches share an object field", () => {
    expect(
      getNestedOutputProperties({
        anyOf: [
          {
            type: "object",
            properties: {
              profile: {
                type: "object",
                title: "Profile",
                properties: {
                  first_name: { type: "string", title: "First Name" },
                },
              },
            },
          },
          {
            type: "object",
            properties: {
              profile: {
                type: "object",
                properties: {
                  last_name: { type: "string", title: "Last Name" },
                },
              },
            },
          },
        ],
      }),
    ).toEqual({
      profile: {
        type: "object",
        title: "Profile",
        properties: {
          first_name: { type: "string", title: "First Name" },
          last_name: { type: "string", title: "Last Name" },
        },
      },
    });
    expect(
      getNestedOutputProperties({
        oneOf: [
          {
            type: "object",
            properties: { value: { type: "string", title: "First" } },
          },
          {
            type: "object",
            properties: {
              value: {
                type: "object",
                properties: { extra: { type: "string" } },
              },
            },
          },
        ],
      }),
    ).toEqual({
      value: { type: "string", title: "First" },
    });
  });

  it("treats prototype-like union keys as own properties", () => {
    const before = Object.getOwnPropertyDescriptors(Object.prototype);
    const schema = JSON.parse(
      '{"anyOf":[{"type":"object","properties":{"__proto__":{"type":"string","title":"Injected"}}},{"type":"null"}]}',
    );
    const result = getNestedOutputProperties(schema);
    expect(Object.getOwnPropertyDescriptors(Object.prototype)).toEqual(before);
    expect(Object.hasOwn(result, "__proto__")).toBe(true);
    expect(result.__proto__).toEqual({ type: "string", title: "Injected" });
    expect(JSON.stringify(schema)).toBe(
      '{"anyOf":[{"type":"object","properties":{"__proto__":{"type":"string","title":"Injected"}}},{"type":"null"}]}',
    );
  });

  it.each([null, undefined, false, 4, "bad", [], {}].map((value) => [value]))(
    "ignores non-string source handles: %j",
    (sourceHandle) => {
      const schema: RJSFSchema = {
        properties: { response: { type: "object" } },
      };
      const edges = [
        { source: "node-a", sourceHandle },
      ] as unknown as CustomEdge[];
      expect(augmentOutputSchemaWithEdges(schema, "node-a", edges)).toEqual(
        schema.properties,
      );
    },
  );
});
