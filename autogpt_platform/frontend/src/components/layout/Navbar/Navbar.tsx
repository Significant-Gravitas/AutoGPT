import { NavbarMainPage } from "./components/NavbarMainPage";
import { getNavbarAccountData } from "./data";

export async function Navbar() {
  const { isLoggedIn } = await getNavbarAccountData();

  return <NavbarMainPage isLoggedIn={isLoggedIn} />;
}
