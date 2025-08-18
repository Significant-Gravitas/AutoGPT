#!/usr/bin/env node

import { execSync } from "child_process";
import * as path from "path";

function getServerUrl(): string {
  const serverUrl =
    process.env.NEXT_PUBLIC_AGPT_SERVER_URL || "http://localhost:8006/api";
  return serverUrl.replace("/api", "");
}

function fetchOpenApiSpec(): void {
  const baseUrl = getServerUrl();
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
