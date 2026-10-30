import { redirect } from "next/navigation";

interface Props {
  searchParams: Promise<{ role?: string }>;
}

// Creating an expert is a dialog on the Team page now. Old links and Otto's
// `/raise?role=` suggestions open it there, with the role carried over.
export default async function RaisePage({ searchParams }: Props) {
  const { role } = await searchParams;
  const query = new URLSearchParams({ raise: "1", ...(role ? { role } : {}) });
  redirect(`/team?${query.toString()}`);
}
