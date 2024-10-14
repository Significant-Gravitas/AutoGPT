import * as React from "react";
import Link from "next/link";

interface NavLink {
  name: string;
  href: string;
}

interface NavbarProps {
  userName: string;
  links: NavLink[];
  activeLink: string;
  onProfileClick: () => void;
}

export const Navbar: React.FC<NavbarProps> = ({
  userName,
  links,
  activeLink,
  onProfileClick,
}) => {
  return (
    <nav className="flex h-[5.5rem] w-screen items-center justify-between border border-black/10 bg-[#f0f0f0] px-10">
      <div className="flex items-center space-x-10">
        {links.map((link) => (
          <div key={link.name} className="relative">
            <Link href={link.href}>
              <div
                className={`text-[${activeLink === link.href ? "#272727" : "#474747"}] font-neue text-2xl font-medium leading-9 tracking-tight`}
              >
                {link.name}
              </div>
            </Link>
            {activeLink === link.href && (
              <div className="absolute bottom-[-30px] left-[-10px] h-1.5 w-full bg-[#282828]" />
            )}
          </div>
        ))}
      </div>
      <div className="flex items-center space-x-5">
        <div
          className="cursor-pointer font-neue text-2xl font-medium leading-9 tracking-tight text-[#474747]"
          onClick={onProfileClick}
        >
          {userName}
        </div>
        <div className="h-10 w-10 cursor-pointer" onClick={onProfileClick}>
          <div className="h-10 w-10 rounded-full border border-[#474747] bg-[#dbdbdb]" />
        </div>
      </div>
    </nav>
  );
};
