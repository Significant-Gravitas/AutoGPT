import type { Meta, StoryObj } from "@storybook/nextjs";
import { useEffect, useRef, useState } from "react";
import { Confetti } from "./Confetti";
import type { ConfettiRef } from "./Confetti";

const meta = {
  title: "Molecules/Confetti",
  component: Confetti,
  parameters: {
    layout: "fullscreen",
  },
  tags: ["autodocs"],
} satisfies Meta<typeof Confetti>;

export default meta;
type Story = StoryObj<typeof meta>;

function BasicCannonDemo() {
  const confettiRef = useRef<ConfettiRef>(null);

  function fire() {
    confettiRef.current?.fire({
      particleCount: 150,
      spread: 100,
      startVelocity: 40,
      scalar: 1.2,
      origin: { x: 0.5, y: 0.5 },
    });
  }

  return (
    <div className="flex h-screen flex-col items-center justify-center gap-6 bg-zinc-50">
      <Confetti ref={confettiRef} manualstart />
      <p className="text-sm text-zinc-500">
        Single burst from center of viewport
      </p>
      <button
        onClick={fire}
        className="rounded-lg bg-purple-500 px-8 py-3 text-base font-medium text-white shadow-md transition-colors hover:bg-purple-600 active:scale-95"
      >
        Fire Cannon
      </button>
    </div>
  );
}

export const BasicCannon: Story = {
  render: () => <BasicCannonDemo />,
};

function RandomDirectionDemo() {
  const confettiRef = useRef<ConfettiRef>(null);

  function fire() {
    confettiRef.current?.fire({
      particleCount: 120,
      angle: Math.random() * 360,
      spread: 80,
      startVelocity: 45,
      scalar: 1.3,
      origin: { x: Math.random(), y: Math.random() * 0.6 + 0.2 },
    });
  }

  return (
    <div className="flex h-screen flex-col items-center justify-center gap-6 bg-zinc-50">
      <Confetti ref={confettiRef} manualstart />
      <p className="text-sm text-zinc-500">
        Random position and angle each click
      </p>
      <button
        onClick={fire}
        className="rounded-lg bg-purple-500 px-8 py-3 text-base font-medium text-white shadow-md transition-colors hover:bg-purple-600 active:scale-95"
      >
        Random Burst
      </button>
    </div>
  );
}

export const RandomDirection: Story = {
  render: () => <RandomDirectionDemo />,
};

function SideCannonsDemo() {
  const confettiRef = useRef<ConfettiRef>(null);

  function fireSideCannons() {
    confettiRef.current?.fire({
      particleCount: 120,
      angle: 60,
      spread: 70,
      startVelocity: 50,
      scalar: 1.2,
      origin: { x: 0, y: 0.5 },
    });
    confettiRef.current?.fire({
      particleCount: 120,
      angle: 120,
      spread: 70,
      startVelocity: 50,
      scalar: 1.2,
      origin: { x: 1, y: 0.5 },
    });
  }

  return (
    <div className="flex h-screen flex-col items-center justify-center gap-6 bg-zinc-50">
      <Confetti ref={confettiRef} manualstart />
      <p className="text-sm text-zinc-500">Dual cannons from left and right</p>
      <button
        onClick={fireSideCannons}
        className="rounded-lg bg-purple-500 px-8 py-3 text-base font-medium text-white shadow-md transition-colors hover:bg-purple-600 active:scale-95"
      >
        Fire Side Cannons
      </button>
    </div>
  );
}

export const SideCannons: Story = {
  render: () => <SideCannonsDemo />,
};

function FireworksDemo() {
  const confettiRef = useRef<ConfettiRef>(null);
  const [active, setActive] = useState(false);
  const rafRef = useRef<number>(0);

  useEffect(() => {
    return () => cancelAnimationFrame(rafRef.current);
  }, []);

  function startFireworks() {
    setActive(true);
    const duration = 3000;
    const end = Date.now() + duration;

    function frame() {
      confettiRef.current?.fire({
        particleCount: 8,
        angle: 60,
        spread: 70,
        startVelocity: 55,
        scalar: 1.2,
        origin: { x: 0, y: Math.random() * 0.4 + 0.4 },
      });
      confettiRef.current?.fire({
        particleCount: 8,
        angle: 120,
        spread: 70,
        startVelocity: 55,
        scalar: 1.2,
        origin: { x: 1, y: Math.random() * 0.4 + 0.4 },
      });

      if (Date.now() < end) {
        rafRef.current = requestAnimationFrame(frame);
      } else {
        setActive(false);
      }
    }

    frame();
  }

  return (
    <div className="flex h-screen flex-col items-center justify-center gap-6 bg-zinc-900">
      <Confetti ref={confettiRef} manualstart />
      <p className="text-sm text-zinc-400">Continuous stream from both sides</p>
      <button
        onClick={startFireworks}
        disabled={active}
        className="rounded-lg bg-purple-500 px-8 py-3 text-base font-medium text-white shadow-md transition-colors hover:bg-purple-600 active:scale-95 disabled:opacity-50"
      >
        {active ? "Fireworks..." : "Launch Fireworks"}
      </button>
    </div>
  );
}

export const Fireworks: Story = {
  render: () => <FireworksDemo />,
};

function CelebrationDemo() {
  const confettiRef = useRef<ConfettiRef>(null);
  const [celebrating, setCelebrating] = useState(false);
  const rafRef = useRef<number>(0);

  useEffect(() => {
    return () => cancelAnimationFrame(rafRef.current);
  }, []);

  function celebrate() {
    setCelebrating(true);

    // Initial big burst from top
    confettiRef.current?.fire({
      particleCount: 200,
      spread: 160,
      startVelocity: 45,
      scalar: 1.4,
      gravity: 0.8,
      origin: { x: 0.5, y: 0.2 },
    });

    // Then continuous side cannons
    const duration = 2500;
    const end = Date.now() + duration;

    function frame() {
      confettiRef.current?.fire({
        particleCount: 5,
        angle: 60,
        spread: 60,
        startVelocity: 50,
        scalar: 1.2,
        origin: { x: 0, y: 0.5 },
      });
      confettiRef.current?.fire({
        particleCount: 5,
        angle: 120,
        spread: 60,
        startVelocity: 50,
        scalar: 1.2,
        origin: { x: 1, y: 0.5 },
      });

      if (Date.now() < end) {
        rafRef.current = requestAnimationFrame(frame);
      } else {
        setCelebrating(false);
      }
    }

    rafRef.current = requestAnimationFrame(frame);
  }

  return (
    <div className="flex h-screen flex-col items-center justify-center gap-4 bg-purple-50">
      <Confetti ref={confettiRef} manualstart />
      <h1 className="font-poppins text-7xl font-medium tracking-tighter text-purple-700">
        Congrats!
      </h1>
      <p className="text-lg text-purple-500">
        AutoGPT purple-themed celebration
      </p>
      <button
        onClick={celebrate}
        disabled={celebrating}
        className="mt-4 rounded-lg bg-purple-500 px-8 py-3 text-base font-medium text-white shadow-md transition-colors hover:bg-purple-600 active:scale-95 disabled:opacity-50"
      >
        {celebrating ? "Celebrating..." : "Celebrate!"}
      </button>
    </div>
  );
}

export const Celebration: Story = {
  render: () => <CelebrationDemo />,
};

function StarsDemo() {
  const confettiRef = useRef<ConfettiRef>(null);

  function fireStars() {
    confettiRef.current?.fire({
      particleCount: 100,
      spread: 180,
      shapes: ["star"],
      scalar: 2,
      startVelocity: 35,
      gravity: 0.6,
      ticks: 300,
      origin: { x: 0.5, y: 0.4 },
    });
  }

  return (
    <div className="flex h-screen flex-col items-center justify-center gap-6 bg-purple-100">
      <Confetti ref={confettiRef} manualstart />
      <p className="text-sm text-purple-400">Large star-shaped confetti</p>
      <button
        onClick={fireStars}
        className="rounded-lg bg-purple-600 px-8 py-3 text-base font-medium text-white shadow-md transition-colors hover:bg-purple-700 active:scale-95"
      >
        Purple Stars
      </button>
    </div>
  );
}

export const Stars: Story = {
  render: () => <StarsDemo />,
};

function FullScreenShowerDemo() {
  const confettiRef = useRef<ConfettiRef>(null);
  const [active, setActive] = useState(false);
  const rafRef = useRef<number>(0);

  useEffect(() => {
    return () => cancelAnimationFrame(rafRef.current);
  }, []);

  function startShower() {
    setActive(true);
    const duration = 4000;
    const end = Date.now() + duration;

    function frame() {
      // Rain from random positions across the top
      confettiRef.current?.fire({
        particleCount: 6,
        angle: 270,
        spread: 40,
        startVelocity: 20,
        scalar: 1.5,
        gravity: 1.2,
        ticks: 400,
        origin: { x: Math.random(), y: 0 },
      });

      if (Date.now() < end) {
        rafRef.current = requestAnimationFrame(frame);
      } else {
        setActive(false);
      }
    }

    frame();
  }

  return (
    <div className="flex h-screen flex-col items-center justify-center gap-6 bg-zinc-50">
      <Confetti ref={confettiRef} manualstart />
      <p className="text-sm text-zinc-500">
        Confetti rain across the full viewport
      </p>
      <button
        onClick={startShower}
        disabled={active}
        className="rounded-lg bg-purple-500 px-8 py-3 text-base font-medium text-white shadow-md transition-colors hover:bg-purple-600 active:scale-95 disabled:opacity-50"
      >
        {active ? "Raining..." : "Confetti Shower"}
      </button>
    </div>
  );
}

export const FullScreenShower: Story = {
  render: () => <FullScreenShowerDemo />,
};
