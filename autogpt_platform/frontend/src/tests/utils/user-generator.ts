import { faker } from "@faker-js/faker";

export function generateUser() {
  return {
    email: faker.internet.email(),
    password: faker.internet.password(),
    name: faker.person.fullName(),
  };
}
