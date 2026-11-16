import { Faker, base, en, faker } from "@faker-js/faker";
import { expect, test } from "vitest";
import { FAKER_SEED } from "../faker-seed";

// Read when the file loads, before any beforeEach: module-scope test data.
const builtAtLoad = faker.string.alphanumeric(16);

test("faker data a test file builds when it loads is the same every run", () => {
  expect(builtAtLoad).toBe(firstSeededValue());
});

test("each test starts from the seeded faker sequence", () => {
  expect(faker.string.alphanumeric(16)).toBe(firstSeededValue());
});

function firstSeededValue() {
  const reference = new Faker({ locale: [en, base] });
  reference.seed(FAKER_SEED);
  return reference.string.alphanumeric(16);
}
