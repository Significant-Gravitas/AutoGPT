import { describe, expect, it } from "vitest";
import {
  coerceCredentialFields,
  buildSiblingInputsFromCredentials,
  coerceExpectedInputs,
  buildExpectedInputsSchema,
  extractInitialValues,
  mergeInputValues,
  checkAllCredentialsComplete,
  getRequiredInputNames,
  checkAllInputsComplete,
  checkCanRun,
  buildRunMessage,
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

  it("handles non-string items in types array", () => {
    const input = {
      cred1: {
        provider: "github",
        types: [123, null, "api_key", undefined],
      },
    };
    const result = coerceCredentialFields(input);
    expect(result.credentialFields).toHaveLength(1);
    const schema = result.credentialFields[0][1] as Record<string, unknown>;
    expect(schema.credentials_types).toEqual(["api_key"]);
  });

  it("skips entries with empty types array", () => {
    const input = {
      cred1: {
        provider: "github",
        types: [],
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

  it("skips non-object values in the credentials map", () => {
    const input = {
      cred1: "string-value",
      cred2: null,
      cred3: 42,
      cred4: {
        discriminator: "url",
        discriminator_values: ["https://ok.com"],
      },
    };
    const result = buildSiblingInputsFromCredentials(input);
    expect(result.url).toBe("https://ok.com");
    expect(Object.keys(result)).toHaveLength(1);
  });

  it("filters non-string discriminator_values", () => {
    const input = {
      cred1: {
        discriminator: "url",
        discriminator_values: [42, "https://valid.com", null],
      },
    };
    const result = buildSiblingInputsFromCredentials(input);
    expect(result.url).toBe("https://valid.com");
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

  it("uses 'unknown' for non-string type field", () => {
    const result = coerceExpectedInputs([{ name: "q", type: 42 }]);
    expect(result[0].type).toBe("unknown");
  });

  it("skips null value", () => {
    const result = coerceExpectedInputs([
      { name: "q", type: "string", value: null },
    ]);
    expect(result[0].value).toBeUndefined();
  });

  it("preserves arbitrary schema keys (format, picker configs, etc.)", () => {
    const result = coerceExpectedInputs([
      {
        name: "spreadsheet",
        type: "object",
        format: "google-drive-picker",
        google_drive_picker_config: {
          multiselect: false,
          allowed_views: ["SPREADSHEETS"],
        },
        auto_credentials: { provider: "google", type: "oauth2" },
      },
    ]);
    expect(result[0].format).toBe("google-drive-picker");
    expect(result[0].google_drive_picker_config).toEqual({
      multiselect: false,
      allowed_views: ["SPREADSHEETS"],
    });
    expect(result[0].auto_credentials).toEqual({
      provider: "google",
      type: "oauth2",
    });
  });

  it("skips undefined extra keys but keeps other extras", () => {
    const result = coerceExpectedInputs([
      {
        name: "q",
        type: "string",
        format: "email",
        extra_hint: undefined,
      },
    ]);
    expect(result[0].format).toBe("email");
    expect(result[0]).not.toHaveProperty("extra_hint");
  });

  it("omits non-string discriminator_values from scopes in coerceCredentialFields", () => {
    const input = {
      cred1: {
        provider: "github",
        types: ["api_key"],
        scopes: ["read", 42, null, "write"],
      },
    };
    const result = coerceCredentialFields(input);
    const schema = result.credentialFields[0][1] as Record<string, unknown>;
    expect(schema.credentials_scopes).toEqual(["read", "write"]);
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

  it("includes description in schema when present", () => {
    const withDesc = [
      {
        name: "q",
        title: "Q",
        type: "string",
        required: false,
        advanced: false,
        description: "A search query",
      },
    ];
    const schema = buildExpectedInputsSchema(withDesc);
    const props = schema!.properties as Record<string, Record<string, unknown>>;
    expect(props.q.description).toBe("A search query");
  });

  it("returns null when all inputs are advanced and showAdvanced is false", () => {
    const advancedOnly = [
      {
        name: "limit",
        title: "Limit",
        type: "int",
        required: false,
        advanced: true,
      },
    ];
    expect(buildExpectedInputsSchema(advancedOnly)).toBeNull();
    expect(buildExpectedInputsSchema(advancedOnly, true)).not.toBeNull();
  });

  it("propagates arbitrary schema keys (e.g. format, picker config) into properties", () => {
    const pickerInput = [
      {
        name: "spreadsheet",
        title: "Spreadsheet",
        type: "object",
        required: true,
        advanced: false,
        format: "google-drive-picker",
        google_drive_picker_config: {
          multiselect: false,
          allowed_views: ["SPREADSHEETS"],
        },
        auto_credentials: { provider: "google" },
      },
    ];
    const schema = buildExpectedInputsSchema(pickerInput);
    const props = schema!.properties as Record<string, Record<string, unknown>>;
    expect(props.spreadsheet.format).toBe("google-drive-picker");
    expect(props.spreadsheet.google_drive_picker_config).toEqual({
      multiselect: false,
      allowed_views: ["SPREADSHEETS"],
    });
    expect(props.spreadsheet.auto_credentials).toEqual({ provider: "google" });
  });

  it("does not leak reserved keys (name, required, advanced) into properties", () => {
    const input = [
      {
        name: "query",
        title: "Query",
        type: "string",
        required: true,
        advanced: false,
      },
    ];
    const schema = buildExpectedInputsSchema(input);
    const props = schema!.properties as Record<string, Record<string, unknown>>;
    expect(props.query).not.toHaveProperty("name");
    expect(props.query).not.toHaveProperty("required");
    expect(props.query).not.toHaveProperty("advanced");
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

describe("mergeInputValues", () => {
  it("returns initial values when prev is empty", () => {
    expect(mergeInputValues({ a: "1" }, {})).toEqual({ a: "1" });
  });

  it("preserves non-empty prev values over initial", () => {
    expect(mergeInputValues({ a: "1", b: "2" }, { a: "override" })).toEqual({
      a: "override",
      b: "2",
    });
  });

  it("skips undefined, null, and empty string from prev", () => {
    expect(
      mergeInputValues(
        { a: "init-a", b: "init-b", c: "init-c" },
        { a: undefined, b: null, c: "" },
      ),
    ).toEqual({ a: "init-a", b: "init-b", c: "init-c" });
  });

  it("adds new keys from prev that are not in initial", () => {
    expect(mergeInputValues({ a: "1" }, { b: "new" })).toEqual({
      a: "1",
      b: "new",
    });
  });

  it("preserves zero and false as valid values from prev", () => {
    expect(mergeInputValues({ a: "1" }, { a: 0, b: false })).toEqual({
      a: 0,
      b: false,
    });
  });
});

describe("checkAllCredentialsComplete", () => {
  it("returns true when all required credentials are present", () => {
    const required = new Set(["cred1", "cred2"]);
    const input = { cred1: { id: "a" }, cred2: { id: "b" } };
    expect(checkAllCredentialsComplete(required, input)).toBe(true);
  });

  it("returns false when a required credential is missing", () => {
    const required = new Set(["cred1", "cred2"]);
    const input = { cred1: { id: "a" } };
    expect(checkAllCredentialsComplete(required, input)).toBe(false);
  });

  it("returns false when a required credential is falsy", () => {
    const required = new Set(["cred1"]);
    const input = { cred1: undefined };
    expect(checkAllCredentialsComplete(required, input)).toBe(false);
  });

  it("returns true when no credentials are required", () => {
    expect(checkAllCredentialsComplete(new Set(), {})).toBe(true);
  });
});

describe("getRequiredInputNames", () => {
  it("returns names of required non-advanced inputs", () => {
    const inputs = [
      {
        name: "a",
        title: "A",
        type: "string",
        required: true,
        advanced: false,
      },
      {
        name: "b",
        title: "B",
        type: "string",
        required: false,
        advanced: false,
      },
      { name: "c", title: "C", type: "string", required: true, advanced: true },
      {
        name: "d",
        title: "D",
        type: "string",
        required: true,
        advanced: false,
      },
    ];
    expect(getRequiredInputNames(inputs)).toEqual(["a", "d"]);
  });

  it("returns empty array when no inputs are required", () => {
    const inputs = [
      {
        name: "a",
        title: "A",
        type: "string",
        required: false,
        advanced: false,
      },
    ];
    expect(getRequiredInputNames(inputs)).toEqual([]);
  });
});

describe("checkAllInputsComplete", () => {
  it("returns true when there are no inputs", () => {
    expect(checkAllInputsComplete([], {})).toBe(true);
  });

  it("returns true when all required inputs have values", () => {
    const inputs = [
      {
        name: "a",
        title: "A",
        type: "string",
        required: true,
        advanced: false,
      },
      {
        name: "b",
        title: "B",
        type: "string",
        required: false,
        advanced: false,
      },
    ];
    expect(checkAllInputsComplete(inputs, { a: "value" })).toBe(true);
  });

  it("returns false when a required input is empty", () => {
    const inputs = [
      {
        name: "a",
        title: "A",
        type: "string",
        required: true,
        advanced: false,
      },
    ];
    expect(checkAllInputsComplete(inputs, { a: "" })).toBe(false);
  });

  it("returns false when a required input is null", () => {
    const inputs = [
      {
        name: "a",
        title: "A",
        type: "string",
        required: true,
        advanced: false,
      },
    ];
    expect(checkAllInputsComplete(inputs, { a: null })).toBe(false);
  });

  it("returns false when a required input is undefined", () => {
    const inputs = [
      {
        name: "a",
        title: "A",
        type: "string",
        required: true,
        advanced: false,
      },
    ];
    expect(checkAllInputsComplete(inputs, {})).toBe(false);
  });

  it("ignores advanced required inputs", () => {
    const inputs = [
      { name: "a", title: "A", type: "string", required: true, advanced: true },
    ];
    expect(checkAllInputsComplete(inputs, {})).toBe(true);
  });

  it("returns true with only optional inputs present", () => {
    const inputs = [
      {
        name: "a",
        title: "A",
        type: "string",
        required: false,
        advanced: false,
      },
    ];
    expect(checkAllInputsComplete(inputs, {})).toBe(true);
  });
});

describe("checkCanRun", () => {
  it("returns true when no credentials needed and inputs complete", () => {
    expect(checkCanRun(false, false, true)).toBe(true);
  });

  it("returns false when credentials needed but not complete", () => {
    expect(checkCanRun(true, false, true)).toBe(false);
  });

  it("returns false when inputs not complete", () => {
    expect(checkCanRun(false, false, false)).toBe(false);
  });

  it("returns true when credentials needed and complete, inputs complete", () => {
    expect(checkCanRun(true, true, true)).toBe(true);
  });

  it("returns false when both credentials and inputs incomplete", () => {
    expect(checkCanRun(true, false, false)).toBe(false);
  });
});

describe("buildRunMessage", () => {
  it("includes credentials message when needsCredentials is true", () => {
    const msg = buildRunMessage(true, false, {});
    expect(msg).toContain("I've configured the required credentials.");
  });

  it("includes inputs when needsInputs is true", () => {
    const msg = buildRunMessage(false, true, { query: "test" });
    expect(msg).toContain("Run with these inputs:");
    expect(msg).toContain('"query": "test"');
  });

  it("filters out empty/null/undefined values from inputs", () => {
    const msg = buildRunMessage(false, true, {
      a: "keep",
      b: "",
      c: null,
      d: undefined,
    });
    expect(msg).toContain('"a": "keep"');
    expect(msg).not.toContain('"b"');
    expect(msg).not.toContain('"c"');
    expect(msg).not.toContain('"d"');
  });

  it("uses retryInstruction when provided and no inputs", () => {
    const msg = buildRunMessage(false, false, {}, "Retry now please.");
    expect(msg).toBe("Retry now please.");
  });

  it("uses default retry message when no retryInstruction", () => {
    const msg = buildRunMessage(false, false, {});
    expect(msg).toBe("Please re-run this step now.");
  });

  it("combines credentials and inputs messages", () => {
    const msg = buildRunMessage(true, true, { key: "val" });
    expect(msg).toContain("I've configured the required credentials.");
    expect(msg).toContain("Run with these inputs:");
  });
});
