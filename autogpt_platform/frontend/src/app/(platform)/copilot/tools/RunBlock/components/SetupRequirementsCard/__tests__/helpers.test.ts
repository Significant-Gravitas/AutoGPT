import { describe, expect, it } from "vitest";
import {
  coerceCredentialFields,
  buildSiblingInputsFromCredentials,
  coerceExpectedInputs,
  buildExpectedInputsSchema,
  extractInitialValues,
} from "../helpers";

describe("coerceCredentialFields", () => {
  it("returns empty results for null input", () => {
    const result = coerceCredentialFields(null);
    expect(result.credentialFields).toEqual([]);
    expect(result.requiredCredentials.size).toBe(0);
  });

  it("returns empty results for non-object input", () => {
    const result = coerceCredentialFields("not-an-object");
    expect(result.credentialFields).toEqual([]);
  });

  it("parses valid credential with api_key type", () => {
    const input = {
      cred1: {
        provider: "github",
        types: ["api_key"],
      },
    };
    const result = coerceCredentialFields(input);
    expect(result.credentialFields).toHaveLength(1);
    expect(result.credentialFields[0][0]).toBe("cred1");
    expect(result.requiredCredentials.has("cred1")).toBe(true);
  });

  it("filters out invalid credential types", () => {
    const input = {
      cred1: {
        provider: "github",
        types: ["invalid_type"],
      },
    };
    const result = coerceCredentialFields(input);
    expect(result.credentialFields).toHaveLength(0);
  });

  it("skips entries without provider", () => {
    const input = {
      cred1: {
        provider: "",
        types: ["api_key"],
      },
    };
    const result = coerceCredentialFields(input);
    expect(result.credentialFields).toHaveLength(0);
  });

  it("includes discriminator when present", () => {
    const input = {
      cred1: {
        provider: "custom",
        types: ["host_scoped"],
        discriminator: "url",
        discriminator_values: ["https://example.com"],
      },
    };
    const result = coerceCredentialFields(input);
    expect(result.credentialFields).toHaveLength(1);
    const schema = result.credentialFields[0][1] as Record<string, unknown>;
    expect(schema.discriminator).toBe("url");
    expect(schema.discriminator_values).toEqual(["https://example.com"]);
  });

  it("includes scopes when present", () => {
    const input = {
      cred1: {
        provider: "google",
        types: ["oauth2"],
        scopes: ["read", "write"],
      },
    };
    const result = coerceCredentialFields(input);
    const schema = result.credentialFields[0][1] as Record<string, unknown>;
    expect(schema.credentials_scopes).toEqual(["read", "write"]);
  });

  it("handles multiple credentials", () => {
    const input = {
      cred1: { provider: "github", types: ["api_key"] },
      cred2: { provider: "google", types: ["oauth2"] },
    };
    const result = coerceCredentialFields(input);
    expect(result.credentialFields).toHaveLength(2);
    expect(result.requiredCredentials.size).toBe(2);
  });

  it("skips non-object values", () => {
    const input = {
      cred1: "invalid",
      cred2: null,
      cred3: { provider: "github", types: ["api_key"] },
    };
    const result = coerceCredentialFields(input);
    expect(result.credentialFields).toHaveLength(1);
  });
});

describe("buildSiblingInputsFromCredentials", () => {
  it("returns empty object for null input", () => {
    expect(buildSiblingInputsFromCredentials(null)).toEqual({});
  });

  it("returns empty object for non-object input", () => {
    expect(buildSiblingInputsFromCredentials("string")).toEqual({});
  });

  it("extracts discriminator values", () => {
    const input = {
      cred1: {
        discriminator: "url",
        discriminator_values: ["https://example.com"],
      },
    };
    const result = buildSiblingInputsFromCredentials(input);
    expect(result.url).toBe("https://example.com");
  });

  it("takes only the first discriminator value", () => {
    const input = {
      cred1: {
        discriminator: "host",
        discriminator_values: ["first.com", "second.com"],
      },
    };
    const result = buildSiblingInputsFromCredentials(input);
    expect(result.host).toBe("first.com");
  });

  it("skips entries without discriminator", () => {
    const input = {
      cred1: { provider: "github" },
    };
    const result = buildSiblingInputsFromCredentials(input);
    expect(Object.keys(result)).toHaveLength(0);
  });

  it("skips entries with empty discriminator_values", () => {
    const input = {
      cred1: { discriminator: "url", discriminator_values: [] },
    };
    const result = buildSiblingInputsFromCredentials(input);
    expect(Object.keys(result)).toHaveLength(0);
  });
});

describe("coerceExpectedInputs", () => {
  it("returns empty array for non-array input", () => {
    expect(coerceExpectedInputs(null)).toEqual([]);
    expect(coerceExpectedInputs("string")).toEqual([]);
  });

  it("parses valid input objects", () => {
    const result = coerceExpectedInputs([
      { name: "query", title: "Search Query", type: "string", required: true },
    ]);
    expect(result).toHaveLength(1);
    expect(result[0].name).toBe("query");
    expect(result[0].title).toBe("Search Query");
    expect(result[0].type).toBe("string");
    expect(result[0].required).toBe(true);
    expect(result[0].advanced).toBe(false);
  });

  it("generates fallback name from index", () => {
    const result = coerceExpectedInputs([{ type: "string" }]);
    expect(result[0].name).toBe("input-0");
    expect(result[0].title).toBe("input-0");
  });

  it("uses name as fallback title", () => {
    const result = coerceExpectedInputs([{ name: "query", type: "string" }]);
    expect(result[0].title).toBe("query");
  });

  it("includes description when present", () => {
    const result = coerceExpectedInputs([
      { name: "q", type: "string", description: "The search query" },
    ]);
    expect(result[0].description).toBe("The search query");
  });

  it("excludes empty description", () => {
    const result = coerceExpectedInputs([
      { name: "q", type: "string", description: "  " },
    ]);
    expect(result[0].description).toBeUndefined();
  });

  it("includes value when present and non-null", () => {
    const result = coerceExpectedInputs([
      { name: "q", type: "string", value: "default" },
    ]);
    expect(result[0].value).toBe("default");
  });

  it("skips non-object array elements", () => {
    const result = coerceExpectedInputs([
      null,
      "string",
      { name: "valid", type: "string" },
    ]);
    expect(result).toHaveLength(1);
    expect(result[0].name).toBe("valid");
  });
});

describe("buildExpectedInputsSchema", () => {
  const inputs = [
    {
      name: "query",
      title: "Query",
      type: "string",
      required: true,
      advanced: false,
    },
    {
      name: "limit",
      title: "Limit",
      type: "int",
      required: false,
      advanced: true,
    },
  ];

  it("returns null for empty inputs", () => {
    expect(buildExpectedInputsSchema([])).toBeNull();
  });

  it("excludes advanced fields by default", () => {
    const schema = buildExpectedInputsSchema(inputs);
    expect(schema).not.toBeNull();
    expect(schema!.properties).toHaveProperty("query");
    expect(schema!.properties).not.toHaveProperty("limit");
  });

  it("includes advanced fields when showAdvanced is true", () => {
    const schema = buildExpectedInputsSchema(inputs, true);
    expect(schema!.properties).toHaveProperty("query");
    expect(schema!.properties).toHaveProperty("limit");
  });

  it("maps types correctly", () => {
    const allTypes = [
      { name: "a", title: "A", type: "str", required: false, advanced: false },
      { name: "b", title: "B", type: "int", required: false, advanced: false },
      {
        name: "c",
        title: "C",
        type: "float",
        required: false,
        advanced: false,
      },
      {
        name: "d",
        title: "D",
        type: "bool",
        required: false,
        advanced: false,
      },
      {
        name: "e",
        title: "E",
        type: "unknown_type",
        required: false,
        advanced: false,
      },
    ];
    const schema = buildExpectedInputsSchema(allTypes);
    const props = schema!.properties as Record<string, Record<string, unknown>>;
    expect(props.a.type).toBe("string");
    expect(props.b.type).toBe("integer");
    expect(props.c.type).toBe("number");
    expect(props.d.type).toBe("boolean");
    expect(props.e.type).toBe("string");
  });

  it("includes required array only for required fields", () => {
    const schema = buildExpectedInputsSchema(inputs);
    expect(schema!.required).toEqual(["query"]);
  });

  it("omits required when no fields are required", () => {
    const optional = [
      {
        name: "q",
        title: "Q",
        type: "string",
        required: false,
        advanced: false,
      },
    ];
    const schema = buildExpectedInputsSchema(optional);
    expect(schema!.required).toBeUndefined();
  });

  it("includes default value from input.value", () => {
    const withDefault = [
      {
        name: "q",
        title: "Q",
        type: "string",
        required: false,
        advanced: false,
        value: "hello",
      },
    ];
    const schema = buildExpectedInputsSchema(withDefault);
    const props = schema!.properties as Record<string, Record<string, unknown>>;
    expect(props.q.default).toBe("hello");
  });
});

describe("extractInitialValues", () => {
  it("returns empty object when no values are set", () => {
    const inputs = [
      {
        name: "q",
        title: "Q",
        type: "string",
        required: false,
        advanced: false,
      },
    ];
    expect(extractInitialValues(inputs)).toEqual({});
  });

  it("extracts values that are present", () => {
    const inputs = [
      {
        name: "q",
        title: "Q",
        type: "string",
        required: false,
        advanced: false,
        value: "hello",
      },
      {
        name: "n",
        title: "N",
        type: "number",
        required: false,
        advanced: false,
        value: 42,
      },
    ];
    expect(extractInitialValues(inputs)).toEqual({ q: "hello", n: 42 });
  });

  it("skips null and undefined values", () => {
    const inputs = [
      {
        name: "a",
        title: "A",
        type: "string",
        required: false,
        advanced: false,
        value: null,
      },
      {
        name: "b",
        title: "B",
        type: "string",
        required: false,
        advanced: false,
      },
    ];
    expect(extractInitialValues(inputs)).toEqual({});
  });
});
