# Ruby on Rails Integration Guide

This guide shows how to integrate AutoGPT's OAuth popup flow into a Ruby on Rails application, enabling your users to connect their credentials and execute agents.

## Prerequisites

- Ruby 3.0+ and Rails 7.0+
- An OAuth client registered with AutoGPT (see [External API Integration Guide](../external-api-integration.md))
- Your `client_id` and `client_secret`

## Installation

Add these gems to your `Gemfile`:

```ruby
# Gemfile
gem 'httparty'        # HTTP client for API requests
gem 'jwt'             # For decoding OIDC tokens (optional)
```

Then run:

```bash
bundle install
```

## Project Structure

```
app/
├── controllers/
│   └── autogpt/
│       ├── oauth_controller.rb      # OAuth flow handlers
│       ├── webhooks_controller.rb   # Webhook receiver
│       └── connect_controller.rb    # Connect popup page
├── services/
│   └── autogpt/
│       ├── client.rb               # API client
│       └── oauth.rb                # OAuth utilities
├── models/
│   └── autogpt_token.rb           # Token storage (optional)
└── views/
    └── autogpt/
        └── connect/
            └── index.html.erb      # Connect page with popup
config/
├── initializers/
│   └── autogpt.rb                  # Configuration
└── routes.rb                       # Route definitions
```

## Step 1: Configuration

Create `config/initializers/autogpt.rb`:

```ruby
# config/initializers/autogpt.rb
module AutoGPT
  class Configuration
    attr_accessor :client_id, :client_secret, :redirect_uri,
                  :webhook_secret, :base_url

    def initialize
      @base_url = ENV.fetch('AUTOGPT_BASE_URL', 'https://platform.agpt.co')
      @client_id = ENV.fetch('AUTOGPT_CLIENT_ID', nil)
      @client_secret = ENV.fetch('AUTOGPT_CLIENT_SECRET', nil)
      @redirect_uri = ENV.fetch('AUTOGPT_REDIRECT_URI', nil)
      @webhook_secret = ENV.fetch('AUTOGPT_WEBHOOK_SECRET', nil)
    end
  end

  class << self
    def configuration
      @configuration ||= Configuration.new
    end

    def configure
      yield(configuration)
    end
  end
end
```

## Step 2: Create OAuth Service

Create `app/services/autogpt/oauth.rb`:

```ruby
# app/services/autogpt/oauth.rb
require 'securerandom'
require 'base64'
require 'openssl'

module AutoGPT
  class OAuth
    class Error < StandardError; end
    class TokenError < Error; end

    def initialize
      @config = AutoGPT.configuration
    end

    # Generate PKCE code verifier
    def generate_code_verifier
      SecureRandom.urlsafe_base64(32)
    end

    # Generate PKCE code challenge from verifier
    def generate_code_challenge(verifier)
      digest = OpenSSL::Digest::SHA256.digest(verifier)
      Base64.urlsafe_encode64(digest, padding: false)
    end

    # Build authorization URL with PKCE
    def authorization_url(state:, code_verifier:, scopes:)
      code_challenge = generate_code_challenge(code_verifier)

      params = {
        response_type: 'code',
        client_id: @config.client_id,
        redirect_uri: @config.redirect_uri,
        state: state,
        code_challenge: code_challenge,
        code_challenge_method: 'S256',
        scope: scopes.join(' ')
      }

      "#{@config.base_url}/oauth/authorize?#{URI.encode_www_form(params)}"
    end

    # Exchange authorization code for tokens
    def exchange_code(code:, code_verifier:)
      response = HTTParty.post(
        "#{@config.base_url}/oauth/token",
        body: {
          grant_type: 'authorization_code',
          code: code,
          redirect_uri: @config.redirect_uri,
          client_id: @config.client_id,
          client_secret: @config.client_secret,
          code_verifier: code_verifier
        },
        headers: { 'Content-Type' => 'application/x-www-form-urlencoded' }
      )

      handle_token_response(response)
    end

    # Refresh an access token
    def refresh_token(refresh_token)
      response = HTTParty.post(
        "#{@config.base_url}/oauth/token",
        body: {
          grant_type: 'refresh_token',
          refresh_token: refresh_token,
          client_id: @config.client_id,
          client_secret: @config.client_secret
        },
        headers: { 'Content-Type' => 'application/x-www-form-urlencoded' }
      )

      handle_token_response(response)
    end

    # Revoke a token
    def revoke_token(token, token_type: 'access_token')
      HTTParty.post(
        "#{@config.base_url}/oauth/revoke",
        body: {
          token: token,
          token_type_hint: token_type,
          client_id: @config.client_id,
          client_secret: @config.client_secret
        },
        headers: { 'Content-Type' => 'application/x-www-form-urlencoded' }
      )
    end

    private

    def handle_token_response(response)
      if response.success?
        body = response.parsed_response
        {
          access_token: body['access_token'],
          refresh_token: body['refresh_token'],
          token_type: body['token_type'],
          expires_in: body['expires_in'],
          expires_at: Time.current + body['expires_in'].to_i.seconds
        }
      else
        error = response.parsed_response
        raise TokenError, error['error_description'] || error['error'] || 'Token request failed'
      end
    end
  end
end
```

## Step 3: Create API Client

Create `app/services/autogpt/client.rb`:

```ruby
# app/services/autogpt/client.rb
module AutoGPT
  class Client
    class Error < StandardError; end
    class AuthenticationError < Error; end
    class RateLimitError < Error; end

    SCOPES = {
      google: %w[
        google:gmail.readonly google:gmail.send
        google:sheets.read google:sheets.write
        google:calendar.read google:calendar.write
        google:drive.read google:drive.write
      ],
      github: %w[github:repo.read github:repo.write github:user.read],
      twitter: %w[twitter:tweet.read twitter:tweet.write twitter:user.read],
      notion: %w[notion:read notion:write],
      slack: %w[slack:read slack:write]
    }.freeze

    def initialize(access_token:, refresh_token: nil, on_token_refresh: nil)
      @config = AutoGPT.configuration
      @access_token = access_token
      @refresh_token = refresh_token
      @on_token_refresh = on_token_refresh
    end

    # Build connect popup URL for requesting credential grants
    def connect_url(provider:, scopes:, nonce:, redirect_origin:)
      params = {
        client_id: @config.client_id,
        scopes: scopes.join(','),
        nonce: nonce,
        redirect_origin: redirect_origin
      }

      "#{@config.base_url}/connect/#{provider}?#{URI.encode_www_form(params)}"
    end

    # Execute an agent
    def execute_agent(agent_id, inputs:, grant_ids: nil, webhook_url: nil)
      body = { inputs: inputs }
      body[:grant_ids] = grant_ids if grant_ids.present?
      body[:webhook_url] = webhook_url if webhook_url.present?

      request(
        :post,
        "/api/external/v1/executions/agents/#{agent_id}/execute",
        body: body.to_json
      )
    end

    # Get execution status
    def get_execution(execution_id)
      request(:get, "/api/external/v1/executions/#{execution_id}")
    end

    # Cancel an execution
    def cancel_execution(execution_id)
      request(:post, "/api/external/v1/executions/#{execution_id}/cancel")
    end

    # List credential grants
    def list_grants
      request(:get, '/api/external/v1/grants/')
    end

    # Get grant details
    def get_grant(grant_id)
      request(:get, "/api/external/v1/grants/#{grant_id}")
    end

    # Delete credential via grant
    def delete_grant_credential(grant_id)
      request(:delete, "/api/external/v1/grants/#{grant_id}/credential")
    end

    # Get available capabilities
    def capabilities
      request(:get, '/api/external/v1/executions/capabilities')
    end

    # Wait for execution to complete with polling
    def wait_for_completion(execution_id, timeout: 300, poll_interval: 2)
      start_time = Time.current

      loop do
        status = get_execution(execution_id)

        case status[:status]
        when 'completed'
          return status
        when 'failed'
          raise Error, status[:error] || 'Execution failed'
        end

        if Time.current - start_time > timeout.seconds
          raise Error, 'Execution timeout'
        end

        sleep poll_interval
        yield status if block_given?
      end
    end

    private

    def request(method, path, body: nil, retry_auth: true)
      headers = {
        'Authorization' => "Bearer #{@access_token}",
        'Content-Type' => 'application/json'
      }

      url = "#{@config.base_url}#{path}"

      response = case method
      when :get
        HTTParty.get(url, headers: headers)
      when :post
        HTTParty.post(url, headers: headers, body: body)
      when :delete
        HTTParty.delete(url, headers: headers)
      end

      handle_response(response, method, path, body, retry_auth)
    end

    def handle_response(response, method, path, body, retry_auth)
      case response.code
      when 200..299
        symbolize_keys(response.parsed_response)
      when 401
        if retry_auth && @refresh_token
          refresh_tokens!
          request(method, path, body: body, retry_auth: false)
        else
          raise AuthenticationError, 'Authentication failed'
        end
      when 429
        retry_after = response.headers['Retry-After']&.to_i || 60
        raise RateLimitError, "Rate limited. Retry after #{retry_after} seconds"
      else
        error = response.parsed_response
        message = error.is_a?(Hash) ? (error['detail'] || error['error']) : error.to_s
        raise Error, message || "Request failed with status #{response.code}"
      end
    end

    def refresh_tokens!
      oauth = AutoGPT::OAuth.new
      tokens = oauth.refresh_token(@refresh_token)

      @access_token = tokens[:access_token]
      @refresh_token = tokens[:refresh_token]

      @on_token_refresh&.call(tokens)
    end

    def symbolize_keys(obj)
      case obj
      when Hash
        obj.transform_keys(&:to_sym).transform_values { |v| symbolize_keys(v) }
      when Array
        obj.map { |v| symbolize_keys(v) }
      else
        obj
      end
    end
  end
end
```

## Step 4: Create OAuth Controller

Create `app/controllers/autogpt/oauth_controller.rb`:

```ruby
# app/controllers/autogpt/oauth_controller.rb
module AutoGPT
  class OauthController < ApplicationController
    # Skip CSRF for callback (state parameter provides protection)
    skip_before_action :verify_authenticity_token, only: [:callback]

    # GET /autogpt/oauth/authorize
    # Initiates OAuth flow by redirecting to AutoGPT
    def authorize
      oauth = AutoGPT::OAuth.new

      # Generate PKCE parameters
      state = SecureRandom.uuid
      code_verifier = oauth.generate_code_verifier

      # Store in session for callback validation
      session[:autogpt_oauth_state] = state
      session[:autogpt_code_verifier] = code_verifier

      # Build authorization URL
      scopes = %w[openid profile email agents:execute integrations:connect integrations:list]
      url = oauth.authorization_url(
        state: state,
        code_verifier: code_verifier,
        scopes: scopes
      )

      redirect_to url, allow_other_host: true
    end

    # GET /autogpt/oauth/callback
    # Handles OAuth callback and exchanges code for tokens
    def callback
      # Check for OAuth errors
      if params[:error]
        return redirect_to connect_path, alert: params[:error_description] || params[:error]
      end

      # Validate state
      unless params[:state] == session[:autogpt_oauth_state]
        return redirect_to connect_path, alert: 'Invalid OAuth state'
      end

      # Exchange code for tokens
      oauth = AutoGPT::OAuth.new
      begin
        tokens = oauth.exchange_code(
          code: params[:code],
          code_verifier: session[:autogpt_code_verifier]
        )

        # Store tokens securely
        # Option 1: Session (simple, shown here)
        session[:autogpt_tokens] = tokens

        # Option 2: Database (better for production)
        # current_user.update!(
        #   autogpt_access_token: tokens[:access_token],
        #   autogpt_refresh_token: tokens[:refresh_token],
        #   autogpt_token_expires_at: tokens[:expires_at]
        # )

        # Clear OAuth state
        session.delete(:autogpt_oauth_state)
        session.delete(:autogpt_code_verifier)

        redirect_to connect_path, notice: 'Successfully connected to AutoGPT'
      rescue AutoGPT::OAuth::TokenError => e
        redirect_to connect_path, alert: "OAuth error: #{e.message}"
      end
    end

    # DELETE /autogpt/oauth/logout
    # Revokes tokens and clears session
    def logout
      tokens = session[:autogpt_tokens]

      if tokens
        oauth = AutoGPT::OAuth.new
        oauth.revoke_token(tokens[:access_token]) rescue nil
        session.delete(:autogpt_tokens)
      end

      redirect_to root_path, notice: 'Disconnected from AutoGPT'
    end

    private

    def connect_path
      autogpt_connect_path
    end
  end
end
```

## Step 5: Create Webhooks Controller

Create `app/controllers/autogpt/webhooks_controller.rb`:

```ruby
# app/controllers/autogpt/webhooks_controller.rb
module AutoGPT
  class WebhooksController < ApplicationController
    skip_before_action :verify_authenticity_token

    # POST /autogpt/webhooks
    def receive
      # Verify webhook signature
      unless verify_signature
        return render json: { error: 'Invalid signature' }, status: :unauthorized
      end

      payload = JSON.parse(request.body.read, symbolize_names: true)

      case payload[:event]
      when 'execution.started'
        handle_execution_started(payload[:data])
      when 'execution.completed'
        handle_execution_completed(payload[:data])
      when 'execution.failed'
        handle_execution_failed(payload[:data])
      when 'grant.revoked'
        handle_grant_revoked(payload[:data])
      else
        Rails.logger.warn "Unknown AutoGPT webhook event: #{payload[:event]}"
      end

      render json: { received: true }
    end

    private

    def verify_signature
      secret = AutoGPT.configuration.webhook_secret
      return true unless secret # Skip verification if no secret configured

      signature = request.headers['X-Webhook-Signature']
      timestamp = request.headers['X-Webhook-Timestamp']

      return false unless signature && timestamp

      # Check timestamp to prevent replay attacks
      timestamp_time = Time.parse(timestamp)
      return false if (Time.current - timestamp_time).abs > 5.minutes

      # Verify HMAC signature
      body = request.body.read
      request.body.rewind

      expected = 'sha256=' + OpenSSL::HMAC.hexdigest('SHA256', secret, body)
      ActiveSupport::SecurityUtils.secure_compare(signature, expected)
    end

    def handle_execution_started(data)
      Rails.logger.info "AutoGPT execution started: #{data[:execution_id]}"

      # Update your database, notify user, etc.
      # Example with ActionCable:
      # ActionCable.server.broadcast(
      #   "user_#{data[:user_id]}",
      #   { type: 'execution_started', execution_id: data[:execution_id] }
      # )
    end

    def handle_execution_completed(data)
      Rails.logger.info "AutoGPT execution completed: #{data[:execution_id]}"
      Rails.logger.info "Outputs: #{data[:outputs]}"

      # Store results, notify user, trigger follow-up actions
    end

    def handle_execution_failed(data)
      Rails.logger.error "AutoGPT execution failed: #{data[:execution_id]}"
      Rails.logger.error "Error: #{data[:error]}"

      # Handle failure, notify user, retry logic
    end

    def handle_grant_revoked(data)
      Rails.logger.info "AutoGPT grant revoked: #{data[:grant_id]}"

      # Update UI, disable features that depend on this grant
    end
  end
end
```

## Step 6: Create Connect Controller

Create `app/controllers/autogpt/connect_controller.rb`:

```ruby
# app/controllers/autogpt/connect_controller.rb
module AutoGPT
  class ConnectController < ApplicationController
    before_action :require_autogpt_tokens, except: [:index]

    # GET /autogpt/connect
    def index
      @tokens = session[:autogpt_tokens]
      @grants = []

      if @tokens
        client = autogpt_client
        @grants = client.list_grants rescue []
      end
    end

    # GET /autogpt/connect/:provider/popup_url
    # Returns the popup URL for JavaScript to open
    def popup_url
      nonce = SecureRandom.uuid
      session["autogpt_connect_nonce_#{params[:provider]}"] = nonce

      client = autogpt_client
      scopes = params[:scopes]&.split(',') || default_scopes_for(params[:provider])

      url = client.connect_url(
        provider: params[:provider],
        scopes: scopes,
        nonce: nonce,
        redirect_origin: request.base_url
      )

      render json: { url: url, nonce: nonce }
    end

    # POST /autogpt/connect/verify
    # Verifies the connect result from the popup
    def verify
      provider = params[:provider]
      nonce = params[:nonce]
      stored_nonce = session["autogpt_connect_nonce_#{provider}"]

      unless nonce == stored_nonce
        return render json: { error: 'Invalid nonce' }, status: :unprocessable_entity
      end

      session.delete("autogpt_connect_nonce_#{provider}")

      # Store the grant for the user
      # current_user.autogpt_grants.create!(
      #   provider: provider,
      #   grant_id: params[:grant_id],
      #   credential_id: params[:credential_id]
      # )

      render json: { success: true }
    end

    private

    def require_autogpt_tokens
      unless session[:autogpt_tokens]
        redirect_to autogpt_oauth_authorize_path
      end
    end

    def autogpt_client
      tokens = session[:autogpt_tokens]
      AutoGPT::Client.new(
        access_token: tokens[:access_token],
        refresh_token: tokens[:refresh_token],
        on_token_refresh: ->(new_tokens) { session[:autogpt_tokens] = new_tokens }
      )
    end

    def default_scopes_for(provider)
      AutoGPT::Client::SCOPES[provider.to_sym] || []
    end
  end
end
```

## Step 7: Create Routes

Update `config/routes.rb`:

```ruby
# config/routes.rb
Rails.application.routes.draw do
  namespace :autogpt do
    # OAuth routes
    get 'oauth/authorize', to: 'oauth#authorize'
    get 'oauth/callback', to: 'oauth#callback'
    delete 'oauth/logout', to: 'oauth#logout'

    # Connect routes
    get 'connect', to: 'connect#index'
    get 'connect/:provider/popup_url', to: 'connect#popup_url'
    post 'connect/verify', to: 'connect#verify'

    # Webhook receiver
    post 'webhooks', to: 'webhooks#receive'
  end
end
```

## Step 8: Create Connect View

Create `app/views/autogpt/connect/index.html.erb`:

```erb
<%# app/views/autogpt/connect/index.html.erb %>
<div class="container mx-auto px-4 py-8">
  <h1 class="text-2xl font-bold mb-6">Connect Your Services</h1>

  <% if flash[:alert] %>
    <div class="mb-6 p-4 bg-red-100 text-red-700 rounded-lg">
      <%= flash[:alert] %>
    </div>
  <% end %>

  <% if flash[:notice] %>
    <div class="mb-6 p-4 bg-green-100 text-green-700 rounded-lg">
      <%= flash[:notice] %>
    </div>
  <% end %>

  <% unless @tokens %>
    <div class="text-center">
      <p class="text-gray-600 mb-6">
        Sign in with AutoGPT to use AI agents with your connected services.
      </p>
      <%= link_to 'Sign in with AutoGPT',
                  autogpt_oauth_authorize_path,
                  class: 'px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700' %>
    </div>
  <% else %>
    <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
      <!-- Google Connection -->
      <div class="p-6 border rounded-lg">
        <h2 class="text-xl font-semibold mb-2">Google</h2>
        <p class="text-gray-600 mb-4">Connect Gmail, Sheets, Calendar, and Drive</p>
        <button
          data-controller="autogpt-connect"
          data-autogpt-connect-provider-value="google"
          data-autogpt-connect-scopes-value="google:gmail.readonly,google:sheets.read"
          class="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
        >
          Connect Google
        </button>
      </div>

      <!-- GitHub Connection -->
      <div class="p-6 border rounded-lg">
        <h2 class="text-xl font-semibold mb-2">GitHub</h2>
        <p class="text-gray-600 mb-4">Access repositories and issues</p>
        <button
          data-controller="autogpt-connect"
          data-autogpt-connect-provider-value="github"
          data-autogpt-connect-scopes-value="github:repo.read,github:user.read"
          class="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
        >
          Connect GitHub
        </button>
      </div>

      <!-- Notion Connection -->
      <div class="p-6 border rounded-lg">
        <h2 class="text-xl font-semibold mb-2">Notion</h2>
        <p class="text-gray-600 mb-4">Read and write Notion pages</p>
        <button
          data-controller="autogpt-connect"
          data-autogpt-connect-provider-value="notion"
          data-autogpt-connect-scopes-value="notion:read,notion:write"
          class="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
        >
          Connect Notion
        </button>
      </div>
    </div>

    <!-- Connected Services -->
    <% if @grants.any? %>
      <div class="mt-8">
        <h2 class="text-xl font-semibold mb-4">Connected Services</h2>
        <ul class="space-y-2">
          <% @grants.each do |grant| %>
            <li class="p-4 bg-green-50 border border-green-200 rounded-lg flex justify-between items-center">
              <div>
                <span class="font-medium"><%= grant[:provider] %></span>
                <span class="text-gray-500 ml-2">Grant ID: <%= grant[:id] %></span>
              </div>
            </li>
          <% end %>
        </ul>
      </div>
    <% end %>

    <div class="mt-8">
      <%= link_to 'Disconnect',
                  autogpt_oauth_logout_path,
                  method: :delete,
                  class: 'text-red-600 hover:text-red-800' %>
    </div>
  <% end %>
</div>

<script>
// Stimulus controller for Connect buttons
// If using Stimulus, create app/javascript/controllers/autogpt_connect_controller.js
// For simplicity, here's a vanilla JS implementation:

document.querySelectorAll('[data-controller="autogpt-connect"]').forEach(button => {
  button.addEventListener('click', async function() {
    const provider = this.dataset.autogptConnectProviderValue;
    const scopes = this.dataset.autogptConnectScopesValue;

    // Get popup URL from server
    const response = await fetch(`/autogpt/connect/${provider}/popup_url?scopes=${scopes}`);
    const { url, nonce } = await response.json();

    // Open popup
    const width = 500;
    const height = 600;
    const left = window.screenX + (window.outerWidth - width) / 2;
    const top = window.screenY + (window.outerHeight - height) / 2;

    const popup = window.open(
      url,
      'AutoGPT Connect',
      `width=${width},height=${height},left=${left},top=${top},popup=true`
    );

    // Listen for result
    const handler = async (event) => {
      if (event.origin !== '<%= AutoGPT.configuration.base_url %>') return;

      const data = event.data;
      if (data?.type !== 'autogpt_connect_result') return;
      if (data?.nonce !== nonce) return;

      window.removeEventListener('message', handler);
      popup.close();

      if (data.success) {
        // Verify with backend
        await fetch('/autogpt/connect/verify', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'X-CSRF-Token': document.querySelector('meta[name="csrf-token"]').content
          },
          body: JSON.stringify({
            provider: provider,
            nonce: nonce,
            grant_id: data.grant_id,
            credential_id: data.credential_id
          })
        });

        window.location.reload();
      } else {
        alert(`Connection failed: ${data.error_description || data.error}`);
      }
    };

    window.addEventListener('message', handler);
  });
});
</script>
```

## Step 9: Create Stimulus Controller (Optional)

If using Hotwire/Stimulus, create `app/javascript/controllers/autogpt_connect_controller.js`:

```javascript
// app/javascript/controllers/autogpt_connect_controller.js
import { Controller } from "@hotwired/stimulus";

export default class extends Controller {
  static values = {
    provider: String,
    scopes: String,
    baseUrl: { type: String, default: "https://platform.agpt.co" },
  };

  async connect() {
    this.element.addEventListener("click", this.handleClick.bind(this));
  }

  async handleClick(event) {
    event.preventDefault();
    this.element.disabled = true;
    this.element.textContent = "Connecting...";

    try {
      // Get popup URL from server
      const response = await fetch(
        `/autogpt/connect/${this.providerValue}/popup_url?scopes=${this.scopesValue}`
      );
      const { url, nonce } = await response.json();

      // Open popup
      const popup = this.openPopup(url);
      if (!popup) {
        throw new Error("Failed to open popup. Please allow popups.");
      }

      // Wait for result
      const result = await this.waitForResult(popup, nonce);

      // Verify with backend
      await this.verifyResult(result);

      window.location.reload();
    } catch (error) {
      alert(`Connection failed: ${error.message}`);
    } finally {
      this.element.disabled = false;
      this.element.textContent = `Connect ${this.providerValue}`;
    }
  }

  openPopup(url) {
    const width = 500;
    const height = 600;
    const left = window.screenX + (window.outerWidth - width) / 2;
    const top = window.screenY + (window.outerHeight - height) / 2;

    return window.open(
      url,
      "AutoGPT Connect",
      `width=${width},height=${height},left=${left},top=${top},popup=true`
    );
  }

  waitForResult(popup, nonce) {
    return new Promise((resolve, reject) => {
      const pollTimer = setInterval(() => {
        if (popup.closed) {
          clearInterval(pollTimer);
          window.removeEventListener("message", handler);
          reject(new Error("Popup was closed"));
        }
      }, 500);

      const handler = (event) => {
        if (event.origin !== this.baseUrlValue) return;

        const data = event.data;
        if (data?.type !== "autogpt_connect_result") return;
        if (data?.nonce !== nonce) return;

        clearInterval(pollTimer);
        window.removeEventListener("message", handler);
        popup.close();

        if (data.success) {
          resolve(data);
        } else {
          reject(new Error(data.error_description || data.error));
        }
      };

      window.addEventListener("message", handler);
    });
  }

  async verifyResult(result) {
    const csrfToken = document.querySelector('meta[name="csrf-token"]').content;

    const response = await fetch("/autogpt/connect/verify", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-CSRF-Token": csrfToken,
      },
      body: JSON.stringify({
        provider: this.providerValue,
        nonce: result.nonce,
        grant_id: result.grant_id,
        credential_id: result.credential_id,
      }),
    });

    if (!response.ok) {
      throw new Error("Failed to verify connection");
    }
  }
}
```

## Step 10: Environment Variables

Add to your environment configuration:

```bash
# .env or config/credentials.yml.enc
AUTOGPT_BASE_URL=https://platform.agpt.co
AUTOGPT_CLIENT_ID=your_client_id_here
AUTOGPT_CLIENT_SECRET=your_client_secret_here
AUTOGPT_REDIRECT_URI=http://localhost:3000/autogpt/oauth/callback
AUTOGPT_WEBHOOK_SECRET=your_webhook_secret_here
```

## Complete Usage Example

Here's a complete example of executing an agent with connected credentials:

```ruby
# app/controllers/agents_controller.rb
class AgentsController < ApplicationController
  before_action :require_autogpt_tokens

  def execute
    client = autogpt_client

    # Execute the agent
    execution = client.execute_agent(
      params[:agent_id],
      inputs: params[:inputs] || {},
      grant_ids: current_user_grants,
      webhook_url: autogpt_webhooks_url
    )

    # Option 1: Return immediately, rely on webhooks
    render json: { execution_id: execution[:execution_id], status: execution[:status] }

    # Option 2: Wait for completion (blocking)
    # result = client.wait_for_completion(execution[:execution_id]) do |status|
    #   Rails.logger.info "Execution status: #{status[:status]}"
    # end
    # render json: result
  end

  def status
    client = autogpt_client
    execution = client.get_execution(params[:execution_id])
    render json: execution
  end

  private

  def require_autogpt_tokens
    unless session[:autogpt_tokens]
      redirect_to autogpt_oauth_authorize_path
    end
  end

  def autogpt_client
    tokens = session[:autogpt_tokens]
    AutoGPT::Client.new(
      access_token: tokens[:access_token],
      refresh_token: tokens[:refresh_token],
      on_token_refresh: ->(new_tokens) { session[:autogpt_tokens] = new_tokens }
    )
  end

  def current_user_grants
    # Return grant IDs for the current user
    # current_user.autogpt_grants.pluck(:grant_id)
    []
  end
end
```

## Background Job Example

For production, execute agents in background jobs:

```ruby
# app/jobs/execute_agent_job.rb
class ExecuteAgentJob < ApplicationJob
  queue_as :default

  def perform(user_id, agent_id, inputs, grant_ids = [])
    user = User.find(user_id)

    client = AutoGPT::Client.new(
      access_token: user.autogpt_access_token,
      refresh_token: user.autogpt_refresh_token,
      on_token_refresh: ->(tokens) { user.update!(autogpt_tokens_from(tokens)) }
    )

    # Execute agent
    execution = client.execute_agent(
      agent_id,
      inputs: inputs,
      grant_ids: grant_ids,
      webhook_url: Rails.application.routes.url_helpers.autogpt_webhooks_url
    )

    # Create execution record
    user.agent_executions.create!(
      execution_id: execution[:execution_id],
      agent_id: agent_id,
      status: execution[:status]
    )
  end
end
```

## Security Best Practices

1. **Store secrets securely** - Use Rails credentials or environment variables
2. **Validate state parameter** - Prevents CSRF attacks
3. **Use PKCE** - Required for all authorization flows
4. **Verify webhook signatures** - Prevents spoofed webhook calls
5. **Verify popup origin** - Only accept messages from `platform.agpt.co`
6. **Use secure session storage** - Consider database-backed sessions for production
7. **Implement token refresh** - Handle expired tokens gracefully
8. **Use background jobs** - Don't block requests waiting for execution

## Next Steps

- [External API Integration Guide](../external-api-integration.md) - Full API reference
- [Next.js Integration](./nextjs.md) - Client-side integration example
- [Discord Community](https://discord.gg/autogpt) - Get help from the community
