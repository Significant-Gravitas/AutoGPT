/**
 * Cloudflare Workers Script for docs.agpt.co → agpt.co/docs migration
 * 
 * Deploy this script to handle all redirects with a single JavaScript file.
 * No rule limits, easy to maintain, handles all edge cases.
 */

// URL mapping for special cases that don't follow patterns
const SPECIAL_MAPPINGS = {
  // Root page
  '/': '/docs/platform',
  
  // Special cases that don't follow standard patterns
  '/platform/d_id/': '/docs/integrations/block-integrations/d-id',
  '/platform/blocks/blocks/': '/docs/integrations',
  '/platform/blocks/decoder_block/': '/docs/integrations/block-integrations/text-decoder',
  '/platform/blocks/http': '/docs/integrations/block-integrations/send-web-request',
  '/platform/blocks/llm/': '/docs/integrations/block-integrations/ai-and-llm',
  '/platform/blocks/time_blocks': '/docs/integrations/block-integrations/time-and-date',
  '/platform/blocks/text_to_speech_block': '/docs/integrations/block-integrations/text-to-speech',
  '/platform/blocks/ai_shortform_video_block': '/docs/integrations/block-integrations/ai-shortform-video',
  '/platform/blocks/replicate_flux_advanced': '/docs/integrations/block-integrations/replicate-flux-advanced',
  '/platform/blocks/flux_kontext': '/docs/integrations/block-integrations/flux-kontext',
  '/platform/blocks/ai_condition/': '/docs/integrations/block-integrations/ai-condition',
  '/platform/blocks/email_block': '/docs/integrations/block-integrations/email',
  '/platform/blocks/google_maps': '/docs/integrations/block-integrations/google-maps',
  '/platform/blocks/google/gmail': '/docs/integrations/block-integrations/gmail',
  '/platform/blocks/github/issues/': '/docs/integrations/block-integrations/github-issues',
  '/platform/blocks/github/repo/': '/docs/integrations/block-integrations/github-repo',
  '/platform/blocks/github/pull_requests': '/docs/integrations/block-integrations/github-pull-requests',
  '/platform/blocks/twitter/twitter': '/docs/integrations/block-integrations/twitter',
  '/classic/setup/': '/docs/classic/setup/setting-up-autogpt-classic',
  '/code-of-conduct/': '/docs/classic/help-us-improve-autogpt/code-of-conduct',
  '/contributing/': '/docs/classic/contributing',
  '/contribute/': '/docs/contribute',
  '/forge/components/introduction/': '/docs/classic/forge/introduction'
};

/**
 * Transform path by replacing underscores with hyphens and removing trailing slashes
 */
function transformPath(path) {
  return path.replace(/_/g, '-').replace(/\/$/, '');
}

/**
 * Handle docs.agpt.co redirects
 */
function handleDocsRedirect(url) {
  const pathname = url.pathname;
  
  // Check special mappings first
  if (SPECIAL_MAPPINGS[pathname]) {
    return `https://agpt.co${SPECIAL_MAPPINGS[pathname]}`;
  }
  
  // Pattern-based redirects
  
  // Platform blocks: /platform/blocks/* → /docs/integrations/block-integrations/*
  if (pathname.startsWith('/platform/blocks/')) {
    const blockName = pathname.substring('/platform/blocks/'.length);
    const transformedName = transformPath(blockName);
    return `https://agpt.co/docs/integrations/block-integrations/${transformedName}`;
  }
  
  // Platform contributing: /platform/contributing/* → /docs/platform/contributing/*
  if (pathname.startsWith('/platform/contributing/')) {
    const subPath = pathname.substring('/platform/contributing/'.length);
    return `https://agpt.co/docs/platform/contributing/${subPath}`;
  }
  
  // Platform general: /platform/* → /docs/platform/* (with underscore→hyphen)
  if (pathname.startsWith('/platform/')) {
    const subPath = pathname.substring('/platform/'.length);
    const transformedPath = transformPath(subPath);
    return `https://agpt.co/docs/platform/${transformedPath}`;
  }
  
  // Forge components: /forge/components/* → /docs/classic/forge/introduction/*
  if (pathname.startsWith('/forge/components/')) {
    const subPath = pathname.substring('/forge/components/'.length);
    return `https://agpt.co/docs/classic/forge/introduction/${subPath}`;
  }
  
  // Forge general: /forge/* → /docs/classic/forge/*
  if (pathname.startsWith('/forge/')) {
    const subPath = pathname.substring('/forge/'.length);
    return `https://agpt.co/docs/classic/forge/${subPath}`;
  }
  
  // Classic: /classic/* → /docs/classic/*
  if (pathname.startsWith('/classic/')) {
    const subPath = pathname.substring('/classic/'.length);
    return `https://agpt.co/docs/classic/${subPath}`;
  }
  
  // Default fallback
  return 'https://agpt.co/docs/';
}

/**
 * Main Worker function
 */
export default {
  async fetch(request, env, ctx) {
    const url = new URL(request.url);
    
    // Only handle docs.agpt.co requests
    if (url.hostname === 'docs.agpt.co') {
      const redirectUrl = handleDocsRedirect(url);
      
      return new Response(null, {
        status: 301,
        headers: {
          'Location': redirectUrl,
          'Cache-Control': 'max-age=300' // Cache redirects for 5 minutes
        }
      });
    }
    
    // For non-docs requests, pass through or return 404
    return new Response('Not Found', { status: 404 });
  }
};

// Test function for local development
export function testRedirects() {
  const testCases = [
    'https://docs.agpt.co/',
    'https://docs.agpt.co/platform/getting-started/',
    'https://docs.agpt.co/platform/advanced_setup/',
    'https://docs.agpt.co/platform/blocks/basic/',
    'https://docs.agpt.co/platform/blocks/ai_condition/',
    'https://docs.agpt.co/classic/setup/',
    'https://docs.agpt.co/forge/components/agents/',
    'https://docs.agpt.co/contributing/',
    'https://docs.agpt.co/unknown-page'
  ];
  
  console.log('Testing redirects:');
  testCases.forEach(testUrl => {
    const url = new URL(testUrl);
    const result = handleDocsRedirect(url);
    console.log(`${testUrl} → ${result}`);
  });
}