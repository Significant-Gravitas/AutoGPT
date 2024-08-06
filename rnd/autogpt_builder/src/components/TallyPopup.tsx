'use client';
import React, { useEffect } from 'react';
import { Button } from './ui/button';

const TallyPopupSimple = () => {
  useEffect(() => {
    // Load Tally script
    const script = document.createElement('script');
    script.src = "https://tally.so/widgets/embed.js";
    script.async = true;
    document.head.appendChild(script);

    return () => {
      document.head.removeChild(script);
    };
  }, []);


  return (
    <>
      <>
      </>
      <div className="fixed bottom-6 right-6 p-3 bg-primary text-primary-foreground shadow-lg transition-all duration-300 ease-in-out z-50">
        <Button
          data-tally-open="3yx2L0"
          data-tally-emoji-text="👋"
          data-tally-emoji-animation="wave"
          data-tally-emoji-size="200"
        >
          Give Feedback
        </Button>

      </div>
    </>
  );
};

export default TallyPopupSimple;
