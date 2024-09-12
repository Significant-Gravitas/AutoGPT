package models

import (
	"fmt"
	"github.com/golang-jwt/jwt/v4"
)

type User struct {
	UserID string `json:"user_id"`
	Email  string `json:"email"`
	Role   string `json:"role"`
}

func NewUserFromPayload(claims jwt.MapClaims) (User, error) {
	userID, ok := claims["sub"].(string)
	if !ok {
		return User{}, fmt.Errorf("invalid or missing 'sub' claim")
	}

	email, _ := claims["email"].(string)
	role, _ := claims["role"].(string)

	return User{
		UserID: userID,
		Email:  email,
		Role:   role,
	}, nil
}
