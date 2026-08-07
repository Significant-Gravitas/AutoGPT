import { notFound } from "next/navigation";
import { TestUiPage } from "./TestUiPage";

export default function Page() {
  if (process.env.NODE_ENV === "production") notFound();
  return <TestUiPage />;
}
