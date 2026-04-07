import { describe, expect, test } from "vitest";
import { setNestedProperty } from "./utils";

const testCases = [
  {
    name: "simple property assignment",
    path: "name",
    value: "John",
    expected: { name: "John" },
  },
  {
    name: "nested property with dot notation",
    path: "user.settings.theme",
    value: "dark",
    expected: { user: { settings: { theme: "dark" } } },
  },
  {
    name: "nested property with slash notation",
    path: "user/settings/language",
    value: "en",
    expected: { user: { settings: { language: "en" } } },
  },
  {
    name: "mixed dot and slash notation",
    path: "user.settings/preferences.color",
    value: "blue",
    expected: { user: { settings: { preferences: { color: "blue" } } } },
  },
  {
    name: "overwrite primitive with object",
    path: "user.details",
    value: { age: 30 },
    expected: { user: { details: { age: 30 } } },
  },
];

describe("setNestedProperty", () => {
  for (const { name, path, value, expected } of testCases) {
    test(name, () => {
      const obj = {};
      setNestedProperty(obj, path, value);
      expect(obj).toEqual(expected);
    });
  }

  test("throws for null object", () => {
    expect(() => {
      setNestedProperty(null, "test", "value");
    }).toThrow("Target must be a non-null object");
  });

  test("throws for undefined object", () => {
    expect(() => {
      setNestedProperty(undefined, "test", "value");
    }).toThrow("Target must be a non-null object");
  });

  test("throws for non-object target", () => {
    expect(() => {
      setNestedProperty("string", "test", "value");
    }).toThrow("Target must be a non-null object");
  });

  test("throws for empty path", () => {
    expect(() => {
      setNestedProperty({}, "", "value");
    }).toThrow("Path must be a non-empty string");
  });

  test("throws for __proto__ access", () => {
    expect(() => {
      setNestedProperty({}, "__proto__.malicious", "attack");
    }).toThrow("Invalid property name: __proto__");
  });

  test("throws for constructor access", () => {
    expect(() => {
      setNestedProperty({}, "constructor.prototype.malicious", "attack");
    }).toThrow("Invalid property name: constructor");
  });

  test("throws for prototype access", () => {
    expect(() => {
      setNestedProperty({}, "obj.prototype.malicious", "attack");
    }).toThrow("Invalid property name: prototype");
  });

  test("prevents prototype pollution", () => {
    const obj = {};

    expect(() => {
      setNestedProperty(obj, "__proto__.polluted", true);
    }).toThrow("Invalid property name: __proto__");

    expect(({} as { polluted?: boolean }).polluted).toBeUndefined();
  });
});
