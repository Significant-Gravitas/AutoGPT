import { useGoogleLogin } from '@react-oauth/google';
import { useState, useCallback } from 'react';

export const useGoogleAuth = (scope: string) => {
  const [token, setToken] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const login = useGoogleLogin({
    onSuccess: (tokenResponse) => {
      setToken(tokenResponse.access_token);
      setError(null);
    },
    onError: (errorResponse) => {
      setError(errorResponse.error_description || 'An error occurred during login');
    },
    scope: scope,
  });

  const logout = useCallback(() => {
    setToken(null);
  }, []);

  return { token, error, login, logout };
};
