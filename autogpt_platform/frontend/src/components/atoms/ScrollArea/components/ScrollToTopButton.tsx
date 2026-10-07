import { Button } from "@/components/atoms/Button/Button";
import { ArrowUp02Icon } from "@hugeicons/core-free-icons";
import { AnimatePresence, motion, useReducedMotion } from "framer-motion";

interface Props {
  visible: boolean;
  onClick: () => void;
}

export function ScrollToTopButton({ visible, onClick }: Props) {
  const reduceMotion = useReducedMotion();
  const hidden = reduceMotion
    ? { opacity: 0 }
    : { opacity: 0, scale: 0.95, y: 8 };
  const shown = reduceMotion ? { opacity: 1 } : { opacity: 1, scale: 1, y: 0 };

  // framer-motion writes `transform` inline, so the centring lives on a
  // static wrapper rather than on the animated element.
  return (
    <div className="pointer-events-none absolute inset-x-0 bottom-6 z-30 flex justify-center">
      <AnimatePresence>
        {visible ? (
          <motion.div
            className="pointer-events-auto"
            initial={hidden}
            animate={shown}
            exit={hidden}
            transition={{ duration: 0.15, ease: [0, 0, 0.2, 1] }}
          >
            <Button
              type="button"
              variant="primary"
              size="icon-lg"
              leadingIcon={ArrowUp02Icon}
              aria-label="Scroll to top"
              className="shadow-md focus-visible:ring-2 focus-visible:ring-zinc-400 focus-visible:ring-offset-2"
              onClick={onClick}
            />
          </motion.div>
        ) : null}
      </AnimatePresence>
    </div>
  );
}
