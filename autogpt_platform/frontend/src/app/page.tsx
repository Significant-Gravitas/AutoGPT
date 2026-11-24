import { redirect } from "next/navigation";

// On the server, so the browser starts /home's load without first running this page.
export default function Page() {
  redirect("/home");
}
