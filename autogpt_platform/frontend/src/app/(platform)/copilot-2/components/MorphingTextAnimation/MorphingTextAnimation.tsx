import { useEffect, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";

const MorphingTextAnimationComponent = ({
  currentText,
}: {
  currentText: string;
}) => {
  const letters = currentText.split("");
  return (
    <motion.span className="inline-flex overflow-hidden">
      {letters.map((char, index) => (
        <motion.span
          key={`${currentText}-${index}`}
          initial={{ opacity: 0, y: 8, rotateX: "80deg", filter: "blur(6px)" }}
          animate={{ opacity: 1, y: 0, rotateX: "0deg", filter: "blur(0px)" }}
          exit={{ opacity: 0, y: -8, rotateX: "-80deg", filter: "blur(6px)" }}
          style={{ willChange: "transform" }}
          transition={{
            delay: 0.015 * index,

            type: "spring",
            bounce: 0.5,
          }}
          className="inline-block"
        >
          {char === " " ? "\u00A0" : char}
        </motion.span>
      ))}
    </motion.span>
  );
};

export const MorphingTextAnimation = () => {
  const textArray = ["Searching for Twitter blocks", "Found 10 twitter blocks"];
  const [currentText, setCurrentText] = useState(textArray[0]);

  useEffect(() => {
    const interval = setInterval(() => {
      setCurrentText(textArray[Math.floor(Math.random() * textArray.length)]);
    }, 1000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div>
      <AnimatePresence mode="popLayout" initial={false}>
        <motion.div
          key={currentText}
          className="whitespace-nowrap text-sm text-muted-foreground"
        >
          <MorphingTextAnimationComponent currentText={currentText} />
        </motion.div>
      </AnimatePresence>
    </div>
  );
};
