"use client";
import { usePathname } from "next/navigation";

interface ProfileColorContainerProps {
  children: React.ReactNode;
}

const PageStructureContainer: React.FC<ProfileColorContainerProps> = ({
  children,
}) => {
  const pathname = usePathname();
  const isProfilePage = pathname?.includes("/profile") || false;

  return (
    <div className={`${isProfilePage ? "bg-zinc-50" : "bg-white"}`}>
      <div className="mx-auto max-w-[1500px]">{children}</div>
    </div>
  );
};

export default PageStructureContainer;
