import { environment } from "@/services/environment";

import { NavbarView } from "./components/NavbarView";
import { getNavbarAccountData } from "./data";

export async function Navbar() {
  const { isLoggedIn } = await getNavbarAccountData();
  const previewBranchName = environment.getPreviewStealingDev();

  return (
    <NavbarView isLoggedIn={isLoggedIn} previewBranchName={previewBranchName} />
  );
}
