// components/RoleBasedAccess.tsx
import React from "react";
import useUser from "@/hooks/useUser";

interface RoleBasedAccessProps {
  allowedRoles: string[];
  children: React.ReactNode;
}

const RoleBasedAccess: React.FC<RoleBasedAccessProps> = ({
  allowedRoles,
  children,
}) => {
  const { role, isLoading } = useUser();

  if (isLoading) {
    return <div>Loading...</div>;
  }

  if (!role || !allowedRoles.includes(role)) {
    return null;
  }

  return <>{children}</>;
};

export default RoleBasedAccess;
