"use client";

import { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react";
import {
	getCurrentUser,
	login as loginRequest,
	logout as logoutRequest,
	register as registerRequest,
	type LoginPayload,
	type RegisterPayload,
	type User,
} from "@/services/AuthService";

type AuthContextValue = {
	user: User | null;
	isLoggedIn: boolean;
	isLoading: boolean;
	login: (payload: LoginPayload) => Promise<void>;
	register: (payload: RegisterPayload) => Promise<void>;
	logout: () => void;
};

const AuthContext = createContext<AuthContextValue | null>(null);

export function AuthProvider({ children }: { children: React.ReactNode }) {
	const [user, setUser] = useState<User | null>(null);
	const [isLoading, setIsLoading] = useState(true);

	useEffect(() => {
		let mounted = true;

		const hydrate = async () => {
			const token = typeof window === "undefined" ? null : localStorage.getItem("access_token");

			if (!token) {
				if (mounted) {
					setUser(null);
					setIsLoading(false);
				}
				return;
			}

			try {
				const currentUser = await getCurrentUser();
				if (mounted) {
					setUser(currentUser);
				}
			} catch {
				logoutRequest();
				if (mounted) {
					setUser(null);
				}
			} finally {
				if (mounted) {
					setIsLoading(false);
				}
			}
		};

		hydrate();

		return () => {
			mounted = false;
		};
	}, []);

	const login = useCallback(async (payload: LoginPayload) => {
		setIsLoading(true);
		try {
			const response = await loginRequest(payload);
			setUser(response.user);
		} finally {
			setIsLoading(false);
		}
	}, []);

	const register = useCallback(async (payload: RegisterPayload) => {
		setIsLoading(true);
		try {
			await registerRequest(payload);
		} finally {
			setIsLoading(false);
		}
	}, []);

	const logout = useCallback(() => {
		logoutRequest();
		setUser(null);
	}, []);

	const value = useMemo<AuthContextValue>(
		() => ({
			user,
			isLoggedIn: user !== null,
			isLoading,
			login,
			register,
			logout,
		}),
		[user, isLoading, login, register, logout]
	);

	return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth(): AuthContextValue {
	const context = useContext(AuthContext);
	if (!context) {
		throw new Error("useAuth must be used within an AuthProvider");
	}

	return context;
}