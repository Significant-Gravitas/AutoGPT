#!/usr/bin/env node

import { getAgptServerBaseUrl } from "@/lib/env-config";
import { execSync } from "child_process";
import * as path from "path";

function fetchOpenApiSpec(): void {
  const baseUrl = getAgptServerBaseUrl();
  const openApiUrl = `${baseUrl}/openapi.json`;
  const outputPath = path.join(
    __dirname,
    "..",
    "src",
    "app",
    "api",
    "openapi.json",
  );

  console.log(`Fetching OpenAPI spec from: ${openApiUrl}`);
  console.log(`Output path: ${outputPath}`);
  console.log(`Current working directory: ${process.cwd()}`);
  console.log(`Script directory (__dirname): ${__dirname}`);

  try {
    // Fetch the OpenAPI spec
    execSync(`curl "${openApiUrl}" > "${outputPath}"`, { stdio: "inherit" });

    // Format with prettier
    execSync(`prettier --write "${outputPath}"`, { stdio: "inherit" });

    console.log("✅ OpenAPI spec fetched and formatted successfully");
  } catch (error) {
    console.error("❌ Failed to fetch OpenAPI spec:", error);
    process.exit(1);
  }
}

if (require.main === module) {
  fetchOpenApiSpec();
}
