import { NavbarView } from "./components/NavbarView";
import { getNavbarAccountData } from "./data";

export async function Navbar() {
  const { isLoggedIn } = await getNavbarAccountData();

  return <NavbarView isLoggedIn={isLoggedIn} />;
}
