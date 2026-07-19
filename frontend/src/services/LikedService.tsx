import { apiRequest } from "./ApiService";
import { unlikeBook } from "./BookService";
import type { Book } from "@/hooks/useBooks";

export type LikedItem = {
	id: number;
	created_at: string;
	book: Book;
};

export async function getLikedBooks(limit = 100): Promise<LikedItem[]> {
	return apiRequest<LikedItem[]>(`/books/liked?limit=${limit}`);
}

export async function deleteLikedBook(bookId: number): Promise<{ id: number; total_likes: number }> {
	return unlikeBook(bookId);
}
