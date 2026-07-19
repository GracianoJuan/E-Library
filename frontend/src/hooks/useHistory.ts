"use client";

import { useState, useCallback } from "react";
import { apiRequest } from "@/services/ApiService";
import type { Book } from "./useBooks";

interface UseHistory {
	books: Book[];
	isLoading: boolean;
	error: string | null;
	fetchHistory: (limit?: number) => Promise<void>;
}

export function useHistory(): UseHistory {
	const [books, setBooks] = useState<Book[]>([]);
	const [isLoading, setIsLoading] = useState(false);
	const [error, setError] = useState<string | null>(null);

	const fetchHistory = useCallback(async (limit = 100) => {
		setIsLoading(true);
		setError(null);
		try {
			const data = await apiRequest<Book[]>(`/history?limit=${limit}`, {
				method: "GET",
			});
			setBooks(data);
		} catch (err) {
			const message =
				err instanceof Error ? err.message : "Failed to fetch history";
			setError(message);
		} finally {
			setIsLoading(false);
		}
	}, []);

	return {
		books,
		isLoading,
		error,
		fetchHistory,
	};
}
