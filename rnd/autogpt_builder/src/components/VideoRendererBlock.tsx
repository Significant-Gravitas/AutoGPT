import React from 'react';

interface VideoRendererProps {
  data: any;
}

const VideoRendererBlock: React.FC<VideoRendererProps> = ({ data }) => {
  // Extract video URL from the correct location in the data structure
  const videoUrl = data.hardcodedValues?.video_url || 
                   (Array.isArray(data.output_data?.video_url) && data.output_data.video_url[0]);

  if (!videoUrl || typeof videoUrl !== 'string') {
    return <div>Invalid or missing video URL</div>;
  }

  const getYouTubeVideoId = (url: string) => {
    const regExp = /^.*((youtu.be\/)|(v\/)|(\/u\/\w\/)|(embed\/)|(watch\?))\??v?=?([^#&?]*).*/;
    const match = url.match(regExp);
    return (match && match[7].length === 11) ? match[7] : null;
  };

  const videoId = getYouTubeVideoId(videoUrl);

  return (
    <div style={{ width: '100%', padding: '10px' }}>
      {videoId ? (
        <iframe
          width="100%"
          height="315"
          src={`https://www.youtube.com/embed/${videoId}`}
          frameBorder="0"
          allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
          allowFullScreen
        ></iframe>
      ) : (
        <video controls width="100%" height="315">
          <source src={videoUrl} type="video/mp4" />
          Your browser does not support the video tag.
        </video>
      )}
    </div>
  );
};

export default VideoRendererBlock;