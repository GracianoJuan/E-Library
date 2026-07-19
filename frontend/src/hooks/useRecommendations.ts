"use client";

import { useState, useCallback } from "react";
import { apiRequest } from "@/services/ApiService";
import type { Book } from "./useBooks";

interface UseRecommendations {
	recommendations: Book[];
	isLoading: boolean;
	error: string | null;
	fetchRecommendations: (limit?: number) => Promise<void>;
	markBookAsRead: (bookId: number) => Promise<void>;
}

export function useRecommendations(): UseRecommendations {
	const [recommendations, setRecommendations] = useState<Book[]>([]);
	const [isLoading, setIsLoading] = useState(false);
	const [error, setError] = useState<string | null>(null);

	const fetchRecommendations = useCallback(async (limit = 15) => {
		setIsLoading(true);
		setError(null);
		try {
			const data = await apiRequest<Book[]>(
				`/recommendations?limit=${limit}`,
				{
					method: "GET",
				}
			);
			setRecommendations(data);
		} catch (err) {
			const message =
				err instanceof Error ? err.message : "Failed to fetch recommendations";
			setError(message);
		} finally {
			setIsLoading(false);
		}
	}, []);

	const markBookAsRead = useCallback(async (bookId: number) => {
		setError(null);
		try {
			await apiRequest<void>(`/books/${bookId}/read`, {
				method: "POST",
			});
			// Refresh recommendations after marking as read
			await fetchRecommendations();
		} catch (err) {
			const message =
				err instanceof Error ? err.message : "Failed to mark book as read";
			setError(message);
			throw err;
		}
	}, [fetchRecommendations]);

	return {
		recommendations,
		isLoading,
		error,
		fetchRecommendations,
		markBookAsRead,
	};
}
