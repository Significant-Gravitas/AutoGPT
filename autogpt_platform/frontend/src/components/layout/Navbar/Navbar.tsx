import { NavbarView } from "./components/NavbarMainPage";
import { getNavbarAccountData } from "./data";

export async function Navbar() {
  const { isLoggedIn } = await getNavbarAccountData();

  return <NavbarView isLoggedIn={isLoggedIn} />;
}
