import { faker } from "@faker-js/faker";

// The generated MSW handlers draw from faker; a fixed seed makes a name
// collision fail every run instead of once in a few hundred.
export const FAKER_SEED = 4504;

export function seedFaker() {
  faker.seed(FAKER_SEED);
}
