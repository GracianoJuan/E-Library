import { apiRequest } from "./ApiService";

export type User = {
	id: number;
	email: string;
	name: string;
};

export type AuthResponse = {
	access_token: string;
	token_type: string;
	user: User;
};

export type RegisterPayload = {
	name: string;
	email: string;
	password: string;
};

export type LoginPayload = {
	email: string;
	password: string;
};

const TOKEN_KEY = "access_token";

export async function register(payload: RegisterPayload): Promise<AuthResponse> {
	return apiRequest<AuthResponse>("/auth/register", {
		method: "POST",
		body: payload,
		withAuth: false,
	});
}

export async function login(payload: LoginPayload): Promise<AuthResponse> {
	const response = await apiRequest<AuthResponse>("/auth/login", {
		method: "POST",
		body: payload,
		withAuth: false,
	});

	// Store token in localStorage
	if (response.access_token) {
		localStorage.setItem(TOKEN_KEY, response.access_token);
	}

	return response;
}

export async function getCurrentUser(): Promise<User> {
	return apiRequest<User>("/auth/me", {
		method: "GET",
	});
}

export function logout(): void {
	localStorage.removeItem(TOKEN_KEY);
}

export function getToken(): string | null {
	if (typeof window === "undefined") return null;
	return localStorage.getItem(TOKEN_KEY);
}

export function isLoggedIn(): boolean {
	if (typeof window === "undefined") return false;
	return !!localStorage.getItem(TOKEN_KEY);
}

// export async function me(): Promise<AuthUser> {
// 	return apiRequest<AuthUser>("/auth/me");
// }
