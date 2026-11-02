// @vitest-environment node
import { readFileSync } from "node:fs";
import {
  diffSchema,
  formatSchemaFinding,
  getExpectedSchema,
} from "@better-auth/core/db/internal";
import type { BetterAuthOptions } from "better-auth";
import { describe, expect, it, vi } from "vitest";

// Better Auth checks the live database against the tables and columns it
// writes when it starts, and refuses every auth request while they differ.
// Prisma owns those tables (backend/schema.prisma and its migrations), so a
// Better Auth upgrade that writes a new column needs a migration first. This
// runs the same comparison against schema.prisma, before anything deploys.

vi.unmock("@/lib/auth/auth");
vi.mock("pg", () => ({ Pool: vi.fn() }));
vi.mock("better-auth", () => ({
  betterAuth: vi.fn((options: BetterAuthOptions) => ({ options })),
}));
vi.mock("better-auth/next-js", () => ({
  nextCookies: vi.fn(() => ({ id: "next-cookies" })),
}));

const schemaPrisma = readFileSync(
  new URL("../../../../../backend/schema.prisma", import.meta.url),
  "utf8",
);

const PRISMA_SCALARS = [
  "String",
  "Boolean",
  "Int",
  "BigInt",
  "Float",
  "Decimal",
  "DateTime",
  "Json",
  "Bytes",
];

// Prisma Client fills these in itself, so the column gets no DEFAULT.
const CLIENT_SIDE_DEFAULT = /@default\((uuid|cuid|nanoid|ulid)\(/;

// Each model's table and the columns its scalar fields create. Relation
// fields have a model type and no column of their own.
function prismaTables(schema: string) {
  const enums = [...schema.matchAll(/^enum\s+(\w+)\s*\{/gm)].map(
    ([, name]) => name,
  );
  const columnTypes = new Set([...PRISMA_SCALARS, ...enums]);
  return [...schema.matchAll(/^model\s+(\w+)\s*\{([\s\S]*?)^\}/gm)].map(
    ([, model, body]) => ({
      name: body.match(/@@map\("([^"]+)"\)/)?.[1] ?? model,
      columns: body.split("\n").flatMap((line) => {
        const field = line.trim().match(/^(\w+)\s+(\w+)(\[\])?(\?)?(.*)$/);
        if (!field || !columnTypes.has(field[2]) || field[3]) return [];
        const [, name, , , optional, attributes] = field;
        return [
          {
            name: attributes.match(/@map\("([^"]+)"\)/)?.[1] ?? name,
            nullable: optional === "?",
            hasDefault:
              attributes.includes("@default(") &&
              !CLIENT_SIDE_DEFAULT.test(attributes),
          },
        ];
      }),
    }),
  );
}

async function problemsWith(schema: string) {
  vi.stubEnv("BETTER_AUTH_SECRET", "test-secret-".padEnd(48, "x"));
  const { auth } = await import("../auth");
  const { options } = auth as unknown as { options: BetterAuthOptions };
  return diffSchema(getExpectedSchema(options), prismaTables(schema)).map(
    (finding) => formatSchemaFinding(finding, "prisma"),
  );
}

describe("the Better Auth tables in schema.prisma", () => {
  it("hold every column Better Auth writes, and require no other", async () => {
    expect(await problemsWith(schemaPrisma)).toEqual([]);
  });

  it("are refused without a column Better Auth writes", async () => {
    const withoutCurve = schemaPrisma.replace(/^\s+crv\s+String\?\n/m, "");

    expect(await problemsWith(withoutCurve)).toEqual([
      expect.stringContaining(
        'Column "crv" is missing from table "UserAuthJwks"',
      ),
    ]);
  });

  it.each([
    ["without a default", "rotatedBy String"],
    [
      "with a default only Prisma Client fills in",
      "rotatedBy String @default(uuid())",
    ],
  ])(
    "are refused with a required column Better Auth never writes, %s",
    async (_, field) => {
      const withExtra = schemaPrisma.replace(
        /^(model AuthJwks \{\n)/m,
        `$1  ${field}\n`,
      );

      expect(await problemsWith(withExtra)).toEqual([
        expect.stringContaining('Column "rotatedBy" on table "UserAuthJwks"'),
      ]);
    },
  );
});
