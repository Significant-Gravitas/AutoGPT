import { Metadata } from "next";
import { LibraryPageContent } from "./LibraryPageContent";

export const metadata: Metadata = {
  title: "Library – AutoGPT Platform",
};

export default function LibraryPage() {
  return <LibraryPageContent />;
}
