import { apiRequest } from "./ApiService";
import type { Book } from "@/hooks/useBooks";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export async function getBook(bookId: number): Promise<Book> {
	return apiRequest<Book>(`/books/${bookId}`);
}

export async function getMostLikedBooks(limit = 15): Promise<Book[]> {
	return apiRequest<Book[]>(`/books/most-liked?limit=${limit}`);
}

export async function getMostReadBooks(limit = 15): Promise<Book[]> {
	return apiRequest<Book[]>(`/books/most-read?limit=${limit}`);
}

export async function getExploreGenres(): Promise<string[]> {
	return apiRequest<string[]>(`/books/genres`);
}

export async function getExploreBooks(params: {
	limit?: number;
	genre?: string;
	category?: string;
	searchField?: "author" | "publisher" | "title";
	query?: string;
} = {}): Promise<Book[]> {
	const searchParams = new URLSearchParams();
	searchParams.set("limit", String(params.limit ?? 12));
	const selectedGenre = params.genre ?? params.category;
	if (selectedGenre && selectedGenre !== "All") searchParams.set("genre", selectedGenre);
	if (params.searchField) searchParams.set("search_field", params.searchField);
	if (params.query) searchParams.set("query", params.query);
	return apiRequest<Book[]>(`/books/explore?${searchParams.toString()}`);
}

export async function likeBook(bookId: number): Promise<{ id: number; total_likes: number }> {
	return apiRequest<{ id: number; total_likes: number }>(`/books/${bookId}/like`, {
		method: "POST",
	});
}

export async function getBookLikeStatus(bookId: number): Promise<{ id: number; is_liked: boolean }> {
	return apiRequest<{ id: number; is_liked: boolean }>(`/books/${bookId}/like`, {
		method: "GET",
	});
}

export async function unlikeBook(bookId: number): Promise<{ id: number; total_likes: number }> {
	return apiRequest<{ id: number; total_likes: number }>(`/books/${bookId}/like`, {
		method: "DELETE",
	});
}

export async function readBook(bookId: number): Promise<{ message: string; history_id: number }> {
	return apiRequest<{ message: string; history_id: number }>(`/books/${bookId}/read`, {
		method: "POST",
	});
}

export async function getBookRecommendations(
	bookId: number,
	params: { limit?: number; threshold?: number } = {}
): Promise<Book[]> {
	const searchParams = new URLSearchParams();
	searchParams.set("limit", String(params.limit ?? 10));
	searchParams.set("threshold", String(params.threshold ?? 0.25));
	return apiRequest<Book[]>(`/books/${bookId}/recommendations?${searchParams.toString()}`);
}

export type BookContentInfo = {
	id: number;
	page_numbers: number[];
};

export async function getBookContentInfo(bookId: number): Promise<BookContentInfo> {
	return apiRequest<BookContentInfo>(`/books/${bookId}/content`);
}

export function getBookContentPageUrl(bookId: number, pageNumber: number): string {
	return `${API_BASE_URL}/books/${bookId}/content/${pageNumber}`;
}

export function getBookPdfUrl(bookId: number): string {
	return `${API_BASE_URL}/books/${bookId}/pdf`;
}