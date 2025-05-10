"use client";
import { usePathname } from "next/navigation";

interface ProfileColorContainerProps {
  children: React.ReactNode;
}

const PageStructureContainer: React.FC<ProfileColorContainerProps> = ({
  children,
}) => {
  const pathname = usePathname();

  const backgroundMap = {
    "/profile": "bg-zinc-50",
    "/library": "bg-gray-100",
  };

  const bgClass =
    Object.entries(backgroundMap).find(([path]) =>
      pathname.includes(path),
    )?.[1] || "bg-white";

  return (
    <div className={bgClass}>
      <div className="mx-auto max-w-[1500px]">{children}</div>
    </div>
  );
};

export default PageStructureContainer;
