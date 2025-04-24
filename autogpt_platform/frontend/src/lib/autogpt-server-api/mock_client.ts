// If the component is inside the storybook and wants to communicate with the backend, it will automatically use this mock client
import { faker } from "@faker-js/faker";
import BackendAPI from "./client";
import {
  Block,
  BlockUIType,
  CredentialsMetaResponse,
  ProfileDetails,
  User,
  UserOnboarding,
} from "./types";

export interface MockClientProps {
  credits?: number;
  blocks?: Block[];
  profile?: ProfileDetails;
  isAuthenticated?: boolean;
}

// Default mock data
export const DEFAULT_MOCK_DATA: Required<MockClientProps> = {
  credits: 10,
  blocks: [
    {
      id: faker.string.uuid(),
      name: faker.lorem.word(),
      description: faker.lorem.sentence(),
      inputSchema: {
        type: "object",
        properties: {},
      },
      outputSchema: {
        type: "object",
        properties: {},
      },
      staticOutput: false,
      categories: [],
      uiType: BlockUIType.STANDARD,
      costs: [],
      hardcodedValues: {},
    },
  ],
  profile: {
    name: "John Doe",
    username: "johndoe",
    description: "",
    links: [],
    avatar_url: "https://avatars.githubusercontent.com/u/123456789?v=4",
  },
  isAuthenticated: true,
};

export default class MockClient extends BackendAPI {
  private props: Required<MockClientProps>;

  constructor(props: MockClientProps = {}) {
    super();
    this.props = {
      ...DEFAULT_MOCK_DATA,
      ...props,
    };
  }

  override isAuthenticated(): Promise<boolean> {
    return Promise.resolve(this.props.isAuthenticated);
  }

  override createUser(): Promise<User> {
    const userId = faker.string.uuid() as unknown as User["id"];
    return Promise.resolve({
      id: userId,
      email: "test@test.com",
    });
  }

  override getUserCredit(page?: string): Promise<{ credits: number }> {
    return Promise.resolve({ credits: this.props.credits });
  }

  override getAutoTopUpConfig(): Promise<{
    amount: number;
    threshold: number;
  }> {
    return Promise.resolve({ amount: 10, threshold: 10 });
  }

  override setAutoTopUpConfig(config: {
    amount: number;
    threshold: number;
  }): Promise<{ amount: number; threshold: number }> {
    return Promise.resolve(config);
  }

  override requestTopUp(amount: number): Promise<{ checkout_url: string }> {
    return Promise.resolve({ checkout_url: "https://checkout.stripe.com" });
  }

  override getUserPaymentPortalLink(): Promise<{ url: string }> {
    return Promise.resolve({ url: "https://payment.stripe.com" });
  }

  override fulfillCheckout(): Promise<void> {
    return Promise.resolve();
  }

  override getBlocks(): Promise<Block[]> {
    return Promise.resolve(this.props.blocks);
  }

  override getStoreProfile(page?: string): Promise<ProfileDetails> {
    console.log("He is mocking in here inside getStoreProfile");
    return Promise.resolve(this.props.profile);
  }

  override isOnboardingEnabled(): Promise<boolean> {
    return Promise.resolve(true);
  }

  override listCredentials(
    provider?: string,
  ): Promise<CredentialsMetaResponse[]> {
    return Promise.resolve([
      {
        id: faker.string.uuid(),
        provider: "openai",
        type: "api_key",
        title: "My OpenAI API Key",
      },
      {
        id: faker.string.uuid(),
        provider: "google",
        type: "oauth2",
        title: "Google Account",
        scopes: ["https://www.googleapis.com/auth/gmail.send"],
        username: "user@example.com",
      },
    ]);
  }

  override getUserOnboarding(): Promise<UserOnboarding> {
    return Promise.resolve({
      completedSteps: [],
      notificationDot: false,
      notified: [],
      rewardedFor: [],
      usageReason: null,
      integrations: [],
      otherIntegrations: null,
      selectedStoreListingVersionId: null,
      agentInput: null,
      onboardingAgentExecutionId: null,
    });
  }
}
