#!/usr/bin/env node

import { getAgptServerBaseUrl } from "@/lib/env-config";
import { execSync } from "child_process";
import * as path from "path";
import * as fs from "fs";

function fetchOpenApiSpec(): void {
  const args = process.argv.slice(2);
  const forceFlag = args.includes("--force");

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

  console.log(`Output path: ${outputPath}`);
  console.log(`Force flag: ${forceFlag}`);

  // Check if local file exists
  const localFileExists = fs.existsSync(outputPath);

  if (!forceFlag && localFileExists) {
    console.log("✅ Using existing local OpenAPI spec file");
    console.log("💡 Use --force flag to fetch from server");
    return;
  }

  if (!localFileExists) {
    console.log("📄 No local OpenAPI spec found, fetching from server...");
  } else {
    console.log(
      "🔄 Force flag detected, fetching fresh OpenAPI spec from server...",
    );
  }

  console.log(`Fetching OpenAPI spec from: ${openApiUrl}`);

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
