"use client";

import { useState, useEffect, useRef } from "react";
import { motion } from "framer-motion";
import { cn } from "@/lib/utils";
import { Icon } from "@phosphor-icons/react";
import { useFavoriteAnimation } from "../../context/FavoriteAnimationContext";

export interface Tab {
  id: string;
  title: string;
  icon: Icon;
}

interface Props {
  tabs: Tab[];
  activeTab: string;
  onTabChange: (tabId: string) => void;
  layoutId?: string;
}

export function LibraryTabs({
  tabs,
  activeTab,
  onTabChange,
  layoutId = "library-tabs",
}: Props) {
  const { registerFavoritesTabRef } = useFavoriteAnimation();

  return (
    <div className="flex items-center gap-2">
      {tabs.map((tab) => (
        <TabButton
          key={tab.id}
          tab={tab}
          isActive={activeTab === tab.id}
          onSelect={onTabChange}
          layoutId={layoutId}
          onRefReady={
            tab.id === "favorites" ? registerFavoritesTabRef : undefined
          }
        />
      ))}
    </div>
  );
}

interface TabButtonProps {
  tab: Tab;
  isActive: boolean;
  onSelect: (tabId: string) => void;
  layoutId: string;
  onRefReady?: (element: HTMLElement | null) => void;
}

function TabButton({
  tab,
  isActive,
  onSelect,
  layoutId,
  onRefReady,
}: TabButtonProps) {
  const [isLoaded, setIsLoaded] = useState(false);
  const buttonRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (isActive && !isLoaded) {
      setIsLoaded(true);
    }
  }, [isActive, isLoaded]);

  useEffect(() => {
    if (onRefReady) {
      onRefReady(buttonRef.current);
    }
  }, [onRefReady]);

  const ButtonIcon = tab.icon;
  const activeColor = "text-primary";

  return (
    <motion.div
      ref={buttonRef}
      layoutId={`${layoutId}-button-${tab.id}`}
      transition={{
        layout: {
          type: "spring",
          damping: 20,
          stiffness: 230,
          mass: 1.2,
          ease: [0.215, 0.61, 0.355, 1],
        },
      }}
      onClick={() => {
        onSelect(tab.id);
        setIsLoaded(true);
      }}
      className="flex h-fit w-fit"
      style={{ willChange: "transform" }}
    >
      <motion.div
        layout
        transition={{
          layout: {
            type: "spring",
            damping: 20,
            stiffness: 230,
            mass: 1.2,
          },
        }}
        className={cn(
          "flex h-fit cursor-pointer items-center gap-1.5 overflow-hidden border border-zinc-200 px-3 py-2 text-black transition-colors duration-75 ease-out hover:border-zinc-300 hover:bg-zinc-300",
          isActive && activeColor,
          isActive ? "px-4" : "px-3",
        )}
        style={{
          borderRadius: "25px",
        }}
      >
        <motion.div
          layoutId={`${layoutId}-icon-${tab.id}`}
          className="shrink-0"
        >
          <ButtonIcon size={18} />
        </motion.div>
        {isActive && (
          <motion.div
            className="flex items-center"
            initial={isLoaded ? { opacity: 0, filter: "blur(4px)" } : false}
            animate={{ opacity: 1, filter: "blur(0px)" }}
            transition={{
              duration: isLoaded ? 0.2 : 0,
              ease: [0.86, 0, 0.07, 1],
            }}
          >
            <motion.span
              layoutId={`${layoutId}-text-${tab.id}`}
              className="font-sans text-sm font-medium text-black"
            >
              {tab.title}
            </motion.span>
          </motion.div>
        )}
      </motion.div>
    </motion.div>
  );
}
