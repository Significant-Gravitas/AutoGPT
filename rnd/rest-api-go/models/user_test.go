package models

import (
	"encoding/json"
	"testing"

	"github.com/golang-jwt/jwt/v4"
	"github.com/stretchr/testify/assert"
)

func TestNewUserFromPayload(t *testing.T) {
	testCases := []struct {
		name          string
		payload       string
		expectedUser  User
		expectedError bool
	}{
		{
			name:    "Valid payload",
			payload: `{"sub": "123", "email": "test@example.com", "role": "user"}`,
			expectedUser: User{
				UserID: "123",
				Email:  "test@example.com",
				Role:   "user",
			},
			expectedError: false,
		},
		{
			name:          "Missing sub claim",
			payload:       `{"email": "test@example.com", "role": "user"}`,
			expectedUser:  User{},
			expectedError: true,
		},
		{
			name:    "Missing optional claims",
			payload: `{"sub": "456"}`,
			expectedUser: User{
				UserID: "456",
				Email:  "",
				Role:   "",
			},
			expectedError: false,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			var claims jwt.MapClaims
			err := json.Unmarshal([]byte(tc.payload), &claims)
			assert.NoError(t, err)

			user, err := NewUserFromPayload(claims)

			if tc.expectedError {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
				assert.Equal(t, tc.expectedUser, user)
			}
		})
	}
}
