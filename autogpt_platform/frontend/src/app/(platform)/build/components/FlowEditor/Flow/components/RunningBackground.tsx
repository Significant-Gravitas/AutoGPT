export const RunningBackground = () => {
  return (
    <div className="absolute inset-0 h-full w-full">
      <style jsx>{`
        @keyframes rotateGradient {
          0% {
            border-image: linear-gradient(
                to right,
                #bc82f3 17%,
                #f5b9ea 24%,
                #8d99ff 35%,
                #aa6eee 58%,
                #ff6778 70%,
                #ffba71 81%,
                #c686ff 92%
              )
              1;
          }
          14.28% {
            border-image: linear-gradient(
                to right,
                #c686ff 17%,
                #bc82f3 24%,
                #f5b9ea 35%,
                #8d99ff 58%,
                #aa6eee 70%,
                #ff6778 81%,
                #ffba71 92%
              )
              1;
          }
          28.56% {
            border-image: linear-gradient(
                to right,
                #ffba71 17%,
                #c686ff 24%,
                #bc82f3 35%,
                #f5b9ea 58%,
                #8d99ff 70%,
                #aa6eee 81%,
                #ff6778 92%
              )
              1;
          }
          42.84% {
            border-image: linear-gradient(
                to right,
                #ff6778 17%,
                #ffba71 24%,
                #c686ff 35%,
                #bc82f3 58%,
                #f5b9ea 70%,
                #8d99ff 81%,
                #aa6eee 92%
              )
              1;
          }
          57.12% {
            border-image: linear-gradient(
                to right,
                #aa6eee 17%,
                #ff6778 24%,
                #ffba71 35%,
                #c686ff 58%,
                #bc82f3 70%,
                #f5b9ea 81%,
                #8d99ff 92%
              )
              1;
          }
          71.4% {
            border-image: linear-gradient(
                to right,
                #8d99ff 17%,
                #aa6eee 24%,
                #ff6778 35%,
                #ffba71 58%,
                #c686ff 70%,
                #bc82f3 81%,
                #f5b9ea 92%
              )
              1;
          }
          85.68% {
            border-image: linear-gradient(
                to right,
                #f5b9ea 17%,
                #8d99ff 24%,
                #aa6eee 35%,
                #ff6778 58%,
                #ffba71 70%,
                #c686ff 81%,
                #bc82f3 92%
              )
              1;
          }
          100% {
            border-image: linear-gradient(
                to right,
                #bc82f3 17%,
                #f5b9ea 24%,
                #8d99ff 35%,
                #aa6eee 58%,
                #ff6778 70%,
                #ffba71 81%,
                #c686ff 92%
              )
              1;
          }
        }
        .animate-gradient {
          animation: rotateGradient 8s linear infinite;
        }
      `}</style>
      <div
        className="animate-gradient absolute inset-0 bg-transparent blur-xl"
        style={{
          borderWidth: "15px",
          borderStyle: "solid",
          borderColor: "transparent",
          borderImage:
            "linear-gradient(to right, #BC82F3 17%, #F5B9EA 24%, #8D99FF 35%, #AA6EEE 58%, #FF6778 70%, #FFBA71 81%, #C686FF 92%) 1",
        }}
      ></div>
      <div
        className="animate-gradient absolute inset-0 bg-transparent blur-lg"
        style={{
          borderWidth: "10px",
          borderStyle: "solid",
          borderColor: "transparent",
          borderImage:
            "linear-gradient(to right, #BC82F3 17%, #F5B9EA 24%, #8D99FF 35%, #AA6EEE 58%, #FF6778 70%, #FFBA71 81%, #C686FF 92%) 1",
        }}
      ></div>
      <div
        className="animate-gradient absolute inset-0 bg-transparent blur-md"
        style={{
          borderWidth: "6px",
          borderStyle: "solid",
          borderColor: "transparent",
          borderImage:
            "linear-gradient(to right, #BC82F3 17%, #F5B9EA 24%, #8D99FF 35%, #AA6EEE 58%, #FF6778 70%, #FFBA71 81%, #C686FF 92%) 1",
        }}
      ></div>
      <div
        className="animate-gradient absolute inset-0 bg-transparent blur-sm"
        style={{
          borderWidth: "6px",
          borderStyle: "solid",
          borderColor: "transparent",
          borderImage:
            "linear-gradient(to right, #BC82F3 17%, #F5B9EA 24%, #8D99FF 35%, #AA6EEE 58%, #FF6778 70%, #FFBA71 81%, #C686FF 92%) 1",
        }}
      ></div>
    </div>
  );
};
