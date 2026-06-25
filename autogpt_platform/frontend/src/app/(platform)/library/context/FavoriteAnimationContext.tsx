"use client";

import {
  createContext,
  useContext,
  useState,
  useCallback,
  useRef,
} from "react";
import { FlyingHeart } from "../components/FlyingHeart/FlyingHeart";

interface FavoriteAnimationContextType {
  triggerFavoriteAnimation: (startPosition: { x: number; y: number }) => void;
  registerFavoritesTabRef: (element: HTMLElement | null) => void;
}

const FavoriteAnimationContext =
  createContext<FavoriteAnimationContextType | null>(null);

interface FavoriteAnimationProviderProps {
  children: React.ReactNode;
  onAnimationComplete?: () => void;
}

export function FavoriteAnimationProvider({
  children,
  onAnimationComplete,
}: FavoriteAnimationProviderProps) {
  const [animationState, setAnimationState] = useState<{
    startPosition: { x: number; y: number } | null;
    targetPosition: { x: number; y: number } | null;
  }>({
    startPosition: null,
    targetPosition: null,
  });

  const favoritesTabRef = useRef<HTMLElement | null>(null);

  const registerFavoritesTabRef = useCallback((element: HTMLElement | null) => {
    favoritesTabRef.current = element;
  }, []);

  const triggerFavoriteAnimation = useCallback(
    (startPosition: { x: number; y: number }) => {
      if (favoritesTabRef.current) {
        const rect = favoritesTabRef.current.getBoundingClientRect();
        const targetPosition = {
          x: rect.left + rect.width / 2 - 12,
          y: rect.top + rect.height / 2 - 12,
        };
        setAnimationState({ startPosition, targetPosition });
      }
    },
    [],
  );

  function handleAnimationComplete() {
    setAnimationState({ startPosition: null, targetPosition: null });
    onAnimationComplete?.();
  }

  return (
    <FavoriteAnimationContext.Provider
      value={{ triggerFavoriteAnimation, registerFavoritesTabRef }}
    >
      {children}
      <FlyingHeart
        startPosition={animationState.startPosition}
        targetPosition={animationState.targetPosition}
        onAnimationComplete={handleAnimationComplete}
      />
    </FavoriteAnimationContext.Provider>
  );
}

export function useFavoriteAnimation() {
  const context = useContext(FavoriteAnimationContext);
  if (!context) {
    throw new Error(
      "useFavoriteAnimation must be used within FavoriteAnimationProvider",
    );
  }
  return context;
}
