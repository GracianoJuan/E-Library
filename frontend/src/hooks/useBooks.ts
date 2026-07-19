"use client";

import { useState, useCallback } from "react";
import { apiRequest } from "@/services/ApiService";

export interface Book {
	id: number;
	title: string;
	author: string;
	description: string;
	isbn: string;
	category?: string;
	genres: string[];
	publisher: string;
	image_url: string;
	total_likes: number;
	total_readers: number;
	content_file?: string | null;
}

interface UseBooks {
	books: Book[];
	isLoading: boolean;
	error: string | null;
	fetchMostLikedBooks: (limit?: number) => Promise<void>;
	fetchBook: (bookId: number) => Promise<Book | null>;
	likeBook: (bookId: number) => Promise<void>;
	unlikeBook: (bookId: number) => Promise<void>;
}

export function useBooks(): UseBooks {
	const [books, setBooks] = useState<Book[]>([]);
	const [isLoading, setIsLoading] = useState(false);
	const [error, setError] = useState<string | null>(null);

	const fetchMostLikedBooks = useCallback(async (limit = 10) => {
		setIsLoading(true);
		setError(null);
		try {
			const data = await apiRequest<Book[]>(`/books/most-liked?limit=${limit}`, {
				method: "GET",
			});
			setBooks(data);
		} catch (err) {
			const message = err instanceof Error ? err.message : "Failed to fetch books";
			setError(message);
		} finally {
			setIsLoading(false);
		}
	}, []);

	const fetchBook = useCallback(async (bookId: number): Promise<Book | null> => {
		setIsLoading(true);
		setError(null);
		try {
			const data = await apiRequest<Book>(`/books/${bookId}`, {
				method: "GET",
			});
			return data;
		} catch (err) {
			const message = err instanceof Error ? err.message : "Failed to fetch book";
			setError(message);
			return null;
		} finally {
			setIsLoading(false);
		}
	}, []);

	const likeBook = useCallback(async (bookId: number) => {
		setError(null);
		try {
			const data = await apiRequest<{ id: number; total_likes: number }>(
				`/books/${bookId}/like`,
				{
					method: "POST",
				}
			);
			// Update book in state if it exists
			setBooks((prevBooks) =>
				prevBooks.map((b) =>
					b.id === bookId ? { ...b, total_likes: data.total_likes } : b
				)
			);
		} catch (err) {
			const message = err instanceof Error ? err.message : "Failed to like book";
			setError(message);
			throw err;
		}
	}, []);

	const unlikeBook = useCallback(async (bookId: number) => {
		setError(null);
		try {
			const data = await apiRequest<{ id: number; total_likes: number }>(
				`/books/${bookId}/like`,
				{
					method: "DELETE",
				}
			);
			// Update book in state if it exists
			setBooks((prevBooks) =>
				prevBooks.map((b) =>
					b.id === bookId ? { ...b, total_likes: data.total_likes } : b
				)
			);
		} catch (err) {
			const message = err instanceof Error ? err.message : "Failed to unlike book";
			setError(message);
			throw err;
		}
	}, []);

	return {
		books,
		isLoading,
		error,
		fetchMostLikedBooks,
		fetchBook,
		likeBook,
		unlikeBook,
	};
}
