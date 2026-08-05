import { notFound } from "next/navigation";
import { ToolUiDebugPage } from "./ToolUiDebugPage";

export default function Page() {
  if (process.env.NODE_ENV === "production") notFound();
  return <ToolUiDebugPage />;
}
