// components/RoleBasedAccess.tsx
import useSupabase from "@/hooks/useSupabase";
import React from "react";

interface RoleBasedAccessProps {
  allowedRoles: string[];
  children: React.ReactNode;
}

const RoleBasedAccess: React.FC<RoleBasedAccessProps> = ({
  allowedRoles,
  children,
}) => {
  const { user, isUserLoading } = useSupabase();

  if (isUserLoading) {
    return <div>Loading...</div>;
  }

  if (!user!.role || !allowedRoles.includes(user!.role)) {
    return null;
  }

  return <>{children}</>;
};

export default RoleBasedAccess;
