-- CreateTable
CREATE TABLE "ProviderRegistry" (
    "name" TEXT NOT NULL,
    "with_oauth" BOOLEAN NOT NULL DEFAULT false,
    "client_id_env_var" TEXT,
    "client_secret_env_var" TEXT,
    "with_api_key" BOOLEAN NOT NULL DEFAULT false,
    "api_key_env_var" TEXT,
    "with_user_password" BOOLEAN NOT NULL DEFAULT false,
    "username_env_var" TEXT,
    "password_env_var" TEXT,

    CONSTRAINT "ProviderRegistry_pkey" PRIMARY KEY ("name")
);
