import React, { useEffect, useRef } from 'react';
import ReactMarkdown from 'react-markdown';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { tomorrow } from 'react-syntax-highlighter/dist/esm/styles/prism';

const ArtifactRenderer = ({ artifactData }) => {
  const iframeRef = useRef(null);

  useEffect(() => {
    console.log("ArtifactRenderer received props:", artifactData);
  }, [artifactData]);

  if (!artifactData) {
    console.error("No artifact data received");
    return <div>No artifact data received</div>;
  }

  // Handle the nested array structure
  const data = Array.isArray(artifactData) && artifactData.length > 0 ? artifactData[0] : artifactData;

  console.log("ArtifactRenderer data:", data);

  if (!data || typeof data !== 'object') {
    console.error("Invalid artifact data structure:", artifactData);
    return <div>Error: Invalid artifact data structure</div>;
  }

  const { type, title, content, language } = data;

  console.log("Extracted data:", { type, title, content, language });

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
        return (
          <iframe
            ref={iframeRef}
            srcDoc={`
              <html>
                <head>
                  <base target="_blank">
                  <style>
                    body { 
                      font-family: Arial, sans-serif; 
                      margin: 0; 
                      padding: 10px; 
                      box-sizing: border-box; 
                      width: 100%; 
                      height: 100%;
                    }
                  </style>
                </head>
                <body>${content}</body>
              </html>
            `}
            style={{width: '100%', border: 'none'}}
            onLoad={() => {
              if (iframeRef.current) {
                iframeRef.current.style.height = `${iframeRef.current.contentWindow.document.body.scrollHeight}px`;
              }
            }}
          />
        );
      case 'image/svg+xml':
        return (
          <div style={{ width: '100%', height: '100%', overflow: 'hidden' }}>
            <div dangerouslySetInnerHTML={{ __html: content }} />
          </div>
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