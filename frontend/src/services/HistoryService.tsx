import { apiRequest } from "./ApiService";
import type { Book } from "@/hooks/useBooks";

export type HistoryItem = {
	id: number;
	read_at: string;
	book: Book;
};

export async function getHistory(limit = 100): Promise<HistoryItem[]> {
	return apiRequest<HistoryItem[]>(`/history?limit=${limit}`);
}

export async function deleteHistory(bookId: number): Promise<{ message: string }> {
	return apiRequest<{ message: string }>(`/history/${bookId}`, {
		method: "DELETE",
	});
}
