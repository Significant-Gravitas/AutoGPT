#!/usr/bin/env node

import { execSync } from "child_process";
import * as path from "path";
import * as fs from "fs";
import * as os from "os";
import { environment } from "@/services/environment";

function fetchOpenApiSpec(): void {
  const args = process.argv.slice(2);
  const forceFlag = args.includes("--force");

  const baseUrl = environment.getAGPTServerBaseUrl();
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

  // Write to a temporary file first to avoid clearing the real file on failure
  const tmpOutputPath = path.join(
    os.tmpdir(),
    `openapi-fetch-${Date.now()}.json`,
  );

  try {
    // Fetch the OpenAPI spec to a temp file
    execSync(`curl "${openApiUrl}" -o "${tmpOutputPath}"`, {
      stdio: "inherit",
    });

    // Format with prettier
    execSync(`prettier --write "${tmpOutputPath}"`, { stdio: "inherit" });

    // Move temp file to final output path
    fs.copyFileSync(tmpOutputPath, outputPath);
    fs.unlinkSync(tmpOutputPath);

    console.log("✅ OpenAPI spec fetched and formatted successfully");
  } catch (error) {
    if (fs.existsSync(tmpOutputPath)) {
      fs.unlinkSync(tmpOutputPath);
    }
    console.error("❌ Failed to fetch OpenAPI spec:", error);
    process.exit(1);
  }
}

if (require.main === module) {
  fetchOpenApiSpec();
}
