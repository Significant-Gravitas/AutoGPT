import { parseWorkspace } from "../src/lib/openui/parse";

async function main() {
  const chunks: Buffer[] = [];
  let bytes = 0;
  for await (const chunk of process.stdin) {
    const data = Buffer.from(chunk);
    bytes += data.length;
    if (bytes > 240_000) throw new Error("Input exceeds the workspace limit.");
    chunks.push(data);
  }
  try {
    parseWorkspace(Buffer.concat(chunks).toString("utf8"));
    process.stdout.write(JSON.stringify({ valid: true, error: "" }));
  } catch (error) {
    process.stdout.write(
      JSON.stringify({
        valid: false,
        error: (error instanceof Error
          ? error.message
          : "Invalid workspace."
        ).slice(0, 2000),
      }),
    );
  }
}

void main().catch(() => {
  process.exitCode = 1;
});
