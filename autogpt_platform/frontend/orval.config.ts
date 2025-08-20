import { defineConfig } from "orval";

export default defineConfig({
  autogpt_api_client: {
    input: {
      target: `./src/app/api/openapi.json`,
      override: {
        transformer: "./src/app/api/transformers/fix-tags.mjs",
      },
    },
    output: {
      workspace: "./src/app/api",
      target: `./__generated__/endpoints`,
      schemas: "./__generated__/models",
      mode: "tags-split",
      client: "react-query",
      httpClient: "fetch",
      indexFiles: false,
      override: {
        mutator: {
          path: "./mutators/custom-mutator.ts",
          name: "customMutator",
        },
        query: {
          useQuery: true,
          useMutation: true,
          usePrefetch: true,
          // Will add more as their use cases arise
        },
        useDates: true,
        operations: {
          "getV2List library agents": {
            query: {
              useInfinite: true,
              useInfiniteQueryParam: "page",
            },
          },
          "getV1List graph executions": {
            query: {
              useInfinite: true,
              useInfiniteQueryParam: "page",
            },
          },
        },
      },
    },
    hooks: {
      afterAllFilesWrite: "prettier --write",
    },
  },
  // autogpt_zod_schema: {
  //   input: {
  //     target: `./src/app/api/openapi.json`,
  //     override: {
  //       transformer: "./src/app/api/transformers/fix-tags.mjs",
  //     },
  //   },
  //   output: {
  //     workspace: "./src/app/api",
  //     target: `./__generated__/zod-schema`,
  //     schemas: "./__generated__/models",
  //     mode: "tags-split",
  //     client: "zod",
  //     indexFiles: false,
  //   },
  //   hooks: {
  //     afterAllFilesWrite: "prettier --write",
  //   },
  // },
});
