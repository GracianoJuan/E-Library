"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";
import { useAuth } from "@/hooks/useAuth";

export function RequireAuth({ children }: { children: React.ReactNode }) {
	const { isLoggedIn, isLoading } = useAuth();
	const router = useRouter();

	useEffect(() => {
		if (!isLoading && !isLoggedIn) {
			router.replace("/login");
		}
	}, [isLoading, isLoggedIn, router]);

	if (isLoading || !isLoggedIn) {
		return (
			<div className="flex min-h-[60vh] items-center justify-center text-sm text-slate-500 dark:text-slate-400">
				Loading...
			</div>
		);
	}

	return <>{children}</>;
}

export function PublicOnly({ children }: { children: React.ReactNode }) {
	const { isLoggedIn, isLoading } = useAuth();
	const router = useRouter();

	useEffect(() => {
		if (!isLoading && isLoggedIn) {
			router.replace("/");
		}
	}, [isLoading, isLoggedIn, router]);

	if (isLoading || isLoggedIn) {
		return (
			<div className="flex min-h-[60vh] items-center justify-center text-sm text-slate-500 dark:text-slate-400">
				Loading...
			</div>
		);
	}

	return <>{children}</>;
}
