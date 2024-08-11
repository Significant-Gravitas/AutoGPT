import React, { useEffect, useRef, useState } from 'react';
import ReactMarkdown from 'react-markdown';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { tomorrow } from 'react-syntax-highlighter/dist/esm/styles/prism';

const ArtifactRenderer = ({ artifactData }) => {
  const iframeRef = useRef(null);
  const [iframeHeight, setIframeHeight] = useState('300px');

  useEffect(() => {
    const data = Array.isArray(artifactData) && artifactData.length > 0 ? artifactData[0] : artifactData;
    if (data && (data.type === 'image/svg+xml' || data.type === 'text/html')) {
      const resizeIframe = () => {
        if (iframeRef.current && iframeRef.current.contentWindow) {
          const height = iframeRef.current.contentWindow.document.body.scrollHeight;
          setIframeHeight(`${height + 20}px`); // Add a small buffer
        }
      };

      // Resize on load and after a short delay (for any dynamic content)
      if (iframeRef.current) {
        iframeRef.current.onload = resizeIframe;
        setTimeout(resizeIframe, 100);
      }
    }
  }, [artifactData]);

  if (!artifactData) {
    console.error("No artifact data received");
    return <div>No artifact data received</div>;
  }

  const data = Array.isArray(artifactData) && artifactData.length > 0 ? artifactData[0] : artifactData;

  if (!data || typeof data !== 'object') {
    console.error("Invalid artifact data structure:", artifactData);
    return <div>Error: Invalid artifact data structure</div>;
  }

  const { type, title, content, language } = data;

  if (!type) {
    console.error("Artifact type is missing:", data);
    return <div>Error: Artifact type is missing</div>;
  }

  const renderContent = () => {
    switch (type) {
      case 'image/png':
      case 'image/jpeg':
      case 'image/gif':
        return <img src={content} alt={title} style={{ maxWidth: '100%', height: 'auto' }} />;
      case 'application/vnd.agpt.code':
        return (
          <SyntaxHighlighter language={language || 'text'} style={tomorrow}>
            {content}
          </SyntaxHighlighter>
        );
      case 'text/markdown':
        return <ReactMarkdown>{content}</ReactMarkdown>;
      case 'text/html':
      case 'image/svg+xml':
        return (
          <iframe
            ref={iframeRef}
            srcDoc={`
              <html>
                <head>
                  <base target="_blank">
                  <style>
                    body {
                      margin: 0;
                      padding: 0;
                      overflow: hidden;
                    }
                    svg, img {
                      max-width: 100%;
                      height: auto;
                    }
                  </style>
                </head>
                <body>${content}</body>
              </html>
            `}
            style={{
              width: '100%',
              height: iframeHeight,
              border: 'none',
              overflow: 'hidden'
            }}
          />
        );
      default:
        return <p>Unsupported artifact type: {type}</p>;
    }
  };

  return (
    <div className="artifact-renderer" style={{ width: '100%', overflow: 'hidden' }}>
      <h3>{title || 'Untitled Artifact'}</h3>
      {renderContent()}
    </div>
  );
};

export default ArtifactRenderer;